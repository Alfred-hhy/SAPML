

use ark_poly_commit::{Polynomial, LabeledPolynomial, PolynomialCommitment, QuerySet, Evaluations, challenge::ChallengeGenerator};
use ark_bls12_377::{Fr, Bls12_377};
use ark_crypto_primitives::sponge::poseidon::{PoseidonSponge, PoseidonConfig};
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::bls12::Bls12;
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;

use ark_ff::{Field, PrimeField, Fp256, BigInteger256, BigInteger};
use ark_poly::domain::radix2::Radix2EvaluationDomain;
use ark_poly::univariate::{DensePolynomial};
use ark_poly_commit::{marlin_pc, DenseUVPolynomial};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use ark_std::rand::SeedableRng;
use std::borrow::Cow;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use ark_ec::Group;
use ark_poly_commit::marlin_pc::UniversalParams;
use clap::{ValueEnum};
use structopt::StructOpt;
use mpc_net::{MpcNet, MpcMultiNet};

use ark_std::{cfg_into_iter, cfg_iter, start_timer, test_rng};
use rayon::prelude::*;

use blake2::digest::generic_array::functional::FunctionalSequence;
use crate::{end_timer, gen_pp};
use log::debug;

use crate::common::{data_dir, FILE_PP, FILE_COM, FILE_DATA, PartyData, get_labeled_poly, get_seeded_rng};
use crate::mpspdz::{parse_shares_from_file, ShareList};
use crate::mpspdz::ShareList::{Sfix, Sint};
use crate::perf_trace_structured::{print_global_stats, print_stats};


struct Opt {
    debug: bool,

    hosts: PathBuf,

    party: u8,

    save: bool,

    precision_f: u32,


    num_args: Option<u64>,
    player_input_binary_path: Option<PathBuf>,
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

pub fn run<E: PrimeField, P: DenseUVPolynomial<E>, PCS: PolynomialCommitment<E, P, PoseidonSponge<E>>>() {
    debug!("Generating public parameters and exchanging commitments");
    let opt = Opt::from_args();
    if opt.debug {
        env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::init();
    }

    MpcMultiNet::init_from_file(opt.hosts.to_str().unwrap(), opt.party as usize);

    let t_gen_commitments = start_timer!(|| "Generate commitments");

    let mut rng = test_rng();
    let mut input_objs: Vec<Vec<E>> = if let Some(filename) = opt.player_input_binary_path {

        let mut filename_format_os = filename.clone().into_os_string();
        filename_format_os.push("-format");
        let filename_format: PathBuf = filename_format_os.into();

        let p = parse_shares_from_file::<PathBuf, f32, i64>(filename, filename_format);
        let inputs_all = p.expect("Failed to parse shares");
        let inputs_int: Vec<Vec<i64>> = cfg_into_iter!(inputs_all).map(|per_obj_list| {
            cfg_into_iter!(per_obj_list).map(|list: ShareList<f32, i64>| -> Vec<i64> {
                match list {
                    Sint(x_list) => {
                        x_list
                    },
                    Sfix(x_list) => {
                        debug!("First 1 value of x_list: {:?}", &x_list[0]);
                        cfg_iter!(x_list).map(|x| {
                            let integer_version = (x * (2u64.pow(opt.precision_f)) as f32).round() as i64;
                            integer_version
                        } ).collect()
                    }
                }
            }).flatten().collect()
        }).collect();

        cfg_iter!(inputs_int).map(|obj_inputs_int| {
            let iter = cfg_iter!(obj_inputs_int);
            print_type_of(&iter);
            iter.map(|x| {
                if *x < 0 {
                    let positive_version = (-1 * x) as u64;
                    -E::from(E::BigInt::from(positive_version))
                } else {
                    E::from(E::BigInt::from(*x as u64))
                }
            }).collect()
        }).collect()
    } else if let Some(num_args) = opt.num_args {
        vec![(0..num_args).map(|_| E::rand(&mut rng)).collect(); 1]
    } else{
        panic!("Either player_input_binary_path or num_args must be specified");
    };
    let len = input_objs.len();
    println!("Number of input objects: {}", len);


    let mut labeled_polys: Vec<LabeledPolynomial<E, P>> = Vec::new();

    input_objs = input_objs.into_iter().map(|mut obj_inputs| {
        obj_inputs.insert(0, E::zero());
        obj_inputs
    }).collect();

    for mut input in input_objs.iter() {

        let secret_poly = P::from_coefficients_slice(input);


        let labeled_poly = get_labeled_poly::<E, P>(secret_poly, None);
        labeled_polys.push(labeled_poly);
    }

    let (ck, vk) = gen_pp::load::<E, P, PCS>();

    debug!("Start");

    let mut compressed_bytes = Vec::new();
    let (labeled_comms, rands) = PCS::commit(&ck, &labeled_polys, Some(&mut rng)).unwrap();

    let comms: Vec<PCS::Commitment> = labeled_comms.into_iter()
        .map(|labeled_comm| labeled_comm.commitment().clone()).collect::<Vec<_>>();

    let num_comms = comms.len() as u8;

    comms.serialize_compressed(&mut compressed_bytes).unwrap();


    println!("Number of bytes in commitment: {}", compressed_bytes.len());

    let all_commitments_bytes = MpcMultiNet::broadcast_bytes_unequal(&compressed_bytes);
    let all_commitments: Vec<Vec<PCS::Commitment>> = all_commitments_bytes.iter().enumerate().map(|(idx, bytes)| {
        println!("Current bytes length: {}", bytes.len());
        let commit_list: Vec<PCS::Commitment> = Vec::<PCS::Commitment>::deserialize_compressed(bytes.as_slice()).expect(format!("Unable to deserialize commitment list for {}", idx).as_str());

        commit_list
    }).collect::<Vec<_>>();

    for (id, masked_poly) in labeled_polys.clone().into_iter().enumerate() {
        let point = E::one();
        let value: E = masked_poly.evaluate(&point);
        println!("Value at point: {}", value);
    }

    if opt.save {
        let party_data: PartyData<E, P, PoseidonSponge<E>, PCS> = PartyData {
            inputs: input_objs,
            party_id: opt.party,
            rands: rands,
            commitments: all_commitments,
        };
        let mut compressed_bytes = Vec::new();
        party_data.serialize_uncompressed(&mut compressed_bytes).unwrap();

        let path = data_dir(&format!("{}_{}", FILE_DATA, opt.party));
        println!("Saving at {:?}", path);
        let mut file = File::create(path).expect("Unable to create file");
        file.write_all(&compressed_bytes).expect("Unable to write data");
    }

    end_timer!(t_gen_commitments);

    print_stats(MpcMultiNet::stats());
    print_global_stats(MpcMultiNet::compute_global_data_sent());
    MpcMultiNet::deinit();
    println!("Done");
}
