

use ark_poly_commit::{Polynomial, LabeledPolynomial, PolynomialCommitment, QuerySet, Evaluations, challenge::ChallengeGenerator, LabeledCommitment};

use ark_crypto_primitives::sponge::poseidon::{PoseidonSponge, PoseidonConfig};
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;

use ark_ff::{Field, PrimeField, Fp256, BigInteger256, BigInteger};
use ark_poly::domain::radix2::Radix2EvaluationDomain;
use ark_poly::univariate::{DensePolynomial};
use ark_poly_commit::{marlin_pc, DenseUVPolynomial};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::SeedableRng;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::ops::{Add, AddAssign};
use std::path::PathBuf;
use ark_ec::Group;
use ark_poly_commit::kzg10::Proof;
use ark_poly_commit::marlin_pc::UniversalParams;
use clap::{ValueEnum};
use structopt::StructOpt;
use mpc_net::{MpcNet, MpcMultiNet};

use ark_std::{cfg_into_iter, start_timer, test_rng};
use crate::{end_timer, gen_pp};
use log::debug;
use rand::prelude::StdRng;
use rand::RngCore;

// mod lib;
use crate::common::{data_dir, FILE_PP, FILE_COM, FILE_DATA, PartyData, get_labeled_poly, get_seeded_rng, test_sponge, AddAssignExt, AddAssignExtRand};

use crate::mpspdz::{parse_evaluations_from_log_file, parse_randomness_from_log_file};
use crate::perf_trace_structured::{print_global_stats, print_stats};


#[cfg(feature = "parallel")]
use rayon::prelude::*;


#[derive(Debug, StructOpt)]
#[structopt(name = "bench", about = "BenchCommit")]
struct Opt {
    #[structopt(short, long)]
    debug: bool,

    #[structopt(long, parse(from_os_str), default_value = "")]
    hosts: PathBuf,

    #[structopt(long, default_value = "0")]
    party: u8,


    #[structopt(long)]
    save: bool,

    #[structopt(long, parse(from_os_str))]
    mpspdz_output_file: PathBuf,
    #[structopt(long)]
    prover_party: Option<u8>
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}


pub fn run<E: PrimeField, P: DenseUVPolynomial<E>, PCS: PolynomialCommitment<E, P, PoseidonSponge<E>>>()

    where <PCS as PolynomialCommitment<E, P, PoseidonSponge<E>>>::Commitment: AddAssignExt, // homomorphic property
     <PCS as PolynomialCommitment<E, P, PoseidonSponge<E>>>::Randomness: for<'a> AddAssignExtRand<&'a <PCS as PolynomialCommitment<E, P, PoseidonSponge<E>>>::Randomness>
{
    debug!("Generating public parameters and exchanging commitments");
    let opt = Opt::from_args();
    if opt.debug {
        env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::init();
    }
    let should_prove = if let Some(prov_party) = &opt.prover_party {
        prov_party == &opt.party
    } else {
        true
    };
    let should_verify = if let Some(prov_party) = &opt.prover_party {
        prov_party != &opt.party
    } else {
        true
    };
    println!("Will prove: {}", should_prove);
    println!("Will verify: {}", should_verify);
    let iter = cfg_into_iter!(Vec::<i32>::new());
    println!("Type of iter:");
    print_type_of(&iter);

    let (ck, vk) = gen_pp::load::<E, P, PCS>();

    let t_prove_verify = start_timer!(|| "Prove Verify");

    let t_read_data = start_timer!(|| "Reading data");

    let party_data: PartyData<E, P, PoseidonSponge<E>, PCS> = load_party_data(&opt);

    let evals: Vec<Vec<(E, E)>> = parse_evaluations_from_log_file(&opt.mpspdz_output_file, party_data.commitments.len()).expect("Unable to parse evaluations");


    let comms = party_data.commitments;
    let rands = party_data.rands;
    end_timer!(t_read_data);

    let mut rng = get_seeded_rng();
    let mut test_sponge = test_sponge::<E>();
    let challenge_generator: ChallengeGenerator<E, PoseidonSponge<E>> = ChallengeGenerator::new_univariate(&mut test_sponge);

    let t_interactive_mpc = start_timer!(|| "Interactive phase");

    MpcMultiNet::init_from_file(opt.hosts.to_str().unwrap(), opt.party as usize);
    debug!("Start");


    let t_build_poly = start_timer!(|| "Building polynomial");
    print!("Building {} polynomials\n", &party_data.inputs.len());

    let mut labeled_polys: Vec<LabeledPolynomial<E, P>> = Vec::new();
    let mut secret_polys: Vec<P> = Vec::new();
    for mut input in party_data.inputs.clone() {
        let secret_poly = P::from_coefficients_slice(&input);
        print!("Building polynomial of size {}\n", &input.len());
        secret_polys.push(secret_poly.clone());

        let labeled_poly = get_labeled_poly::<E, P>(secret_poly, None);
        labeled_polys.push(labeled_poly);
    }

    end_timer!(t_build_poly);

    let (r_commitments, random_polys, r_rands): (Vec<Vec<PCS::Commitment>>, Vec<P>, Vec<PCS::Randomness>) = compute_random_commitment_all::<E, P, PCS>(&ck, &mut rng, &opt, should_prove);

    let (masked_commitments, compressed_proof_bytes) = build_and_open_poly::<E, P, PCS>(&opt, &evals, &ck, &vk, &comms, rands, &mut rng, &challenge_generator, secret_polys, r_commitments, random_polys, &r_rands);

    let t_broadcast_proof = start_timer!(|| "Broadcasting proof");
    let all_proofs_bytes = MpcMultiNet::broadcast_bytes_unequal(&compressed_proof_bytes);
    let all_proofs = all_proofs_bytes.iter().enumerate().map(|(idx, bytes)| {

        let batch_proof = PCS::BatchProof::deserialize_compressed(bytes.as_slice())
            .expect(format!("Unable to witness polynomial {}", idx).as_str());
        return batch_proof
    }).collect::<Vec<_>>();
    end_timer!(t_broadcast_proof);

    let t_verify = start_timer!(|| "Verifying proofs");

    let mut results = all_proofs.iter().enumerate().map(|(party_id, wp)| {
        if party_id == opt.party as usize {
            debug!("Skipping verification for own proof!");
            return true;
        }
        let eval_for_this_party: &Vec<(E, E)> = evals.get(party_id as usize).expect("Something wrong with evals vec");
        if eval_for_this_party.is_empty() {
            println!("No point to verify, skipping verification for party {}", party_id);
            return true;
        }
        println!("Checking {} evaluations for party {} (me: {})", eval_for_this_party.clone().len(), party_id, opt.party);


        let queryset = compute_query_set(&eval_for_this_party.iter().map(|x| x.0).collect());
        let evaluations = compute_evaluations(eval_for_this_party);
        let result = PCS::batch_check(&vk, &masked_commitments[party_id.clone() as usize], &queryset, &evaluations, wp,
                         &mut (challenge_generator.clone()), &mut rng).unwrap();

        if !result {
            println!("Verification failed for party {}", party_id);
            for eval in eval_for_this_party.iter() {
                println!("Eval at point {} is {}", eval.0, eval.1);
            }
        }
        return result;
    });
    if results.all(|x| x) {
        println!("All verifications succeeded!");
    } else {
        println!("Not all verifications succeeded.");
    }
    end_timer!(t_verify);

    end_timer!(t_interactive_mpc);
    end_timer!(t_prove_verify);

    print_stats(MpcMultiNet::stats());
    print_global_stats(MpcMultiNet::compute_global_data_sent());
    MpcMultiNet::deinit();
    println!("Done");
}

fn load_party_data<E: PrimeField, P: DenseUVPolynomial<E>, PCS: PolynomialCommitment<E, P, PoseidonSponge<E>>>(opt: &Opt) -> PartyData<E, P, PoseidonSponge<E>, PCS> {
    let data_file = data_dir(&format!("{}_{}", FILE_DATA, &opt.party));
    debug!("Loading file: {}", data_file.to_str().unwrap());
    if !data_file.exists() {
        println!("Data file does not exist. Run gen_commitments first.");
        panic!();
    }
    let mut file = File::open(data_file).expect("Unable to open file");
    let mut compressed_bytes = Vec::new();
    file.read_to_end(&mut compressed_bytes).expect("Unable to read data");
    let party_data: PartyData<E, P, PoseidonSponge<E>, PCS> = PartyData::deserialize_uncompressed_unchecked(compressed_bytes.as_slice()).expect("Unable to deserialize party data");
    party_data
}

fn build_and_open_poly<E: PrimeField, P: DenseUVPolynomial<E>, PCS: PolynomialCommitment<E, P, PoseidonSponge<E>>>(opt: &Opt, evals: &Vec<Vec<(E, E)>>, ck: &PCS::CommitterKey, vk: &PCS::VerifierKey, comms: &Vec<Vec<PCS::Commitment>>, rands: Vec<PCS::Randomness>, mut rng: &mut StdRng, challenge_generator: &ChallengeGenerator<E, PoseidonSponge<E>>, secret_polys: Vec<P>, r_commitments: Vec<Vec<PCS::Commitment>>, random_polys: Vec<P>, r_rand: &Vec<PCS::Randomness>)
    -> (Vec<Vec<LabeledCommitment<PCS::Commitment>>>, Vec<u8>)
    where <PCS as PolynomialCommitment<E, P, PoseidonSponge<E>>>::Commitment: AddAssignExt,
          <PCS as PolynomialCommitment<E, P, PoseidonSponge<E>>>::Randomness: for<'a> AddAssignExtRand<&'a <PCS as PolynomialCommitment<E, P, PoseidonSponge<E>>>::Randomness>
{

    let masked_polys = secret_polys.into_iter().zip(random_polys.clone().into_iter())
        .enumerate()
        .map(|(id, (secret_poly, random_poly))| {
            let masked_poly = secret_poly.clone() + random_poly.clone();

            let masked_labeled_poly = get_labeled_poly::<E, P>(masked_poly, Some(get_poly_label(id).as_str()));

            masked_labeled_poly
    }).collect::<Vec<_>>();

    let masked_commitments = comms.into_iter()
        .zip(r_commitments.into_iter())
        .map(|(input_com_party, r_com_party)| {
            input_com_party.into_iter().zip(r_com_party)
                .enumerate()
                .map(|(id, (input_com, r_com))| {
                let mut res = input_com.clone();
                res.add_assign_ext(r_com);
                return LabeledCommitment::new(
                    get_poly_label(id),
                    res,
                    None,
                )
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();

    let masked_rands = rands.into_iter().zip(r_rand).map(|(r, r_rand)| {
        let mut res = r.clone();
        res.add_assign_ext(r_rand);
        res
    }).collect::<Vec<_>>();


    let points_opt = evals.get(opt.party as usize).expect("Party not found");
    if points_opt.is_empty() {
        let mut empty_proof_bytes = Vec::new();
        let proofs: Vec<PCS::Proof> = Vec::new();
        let batchproof: PCS::BatchProof = PCS::BatchProof::from(proofs);
        batchproof.serialize_compressed(&mut empty_proof_bytes).unwrap();
        return (masked_commitments, empty_proof_bytes);
    }
    let points = points_opt.iter().map(|p| p.0).collect();

    let t_proof = start_timer!(|| "Opening at points");
    let queryset = compute_query_set(&points);

    let proofs_batched = PCS::batch_open(&ck, &masked_polys,
                                         &masked_commitments[opt.party as usize], &queryset,
                                         &mut (challenge_generator.clone()), &masked_rands, Some(&mut rng)).unwrap();


    let mut compressed_proof_bytes = Vec::new();
    proofs_batched.serialize_compressed(&mut compressed_proof_bytes).unwrap();
    end_timer!(t_proof);
    for (id, masked_poly) in masked_polys.clone().into_iter().enumerate() {
        let value: E = masked_poly.evaluate(&points[id]);
        println!("Prover value at point {} ({}): {}", points[id], id, value);
        if let Some(eval) = evals.get(opt.party as usize) {
            let eval_log = eval.get(id).expect("Point not found").1;

            let random_poly_eval = random_polys[id].evaluate(&E::one());
            println!("Random poly eval {}", random_poly_eval);

        }
    }

    (masked_commitments, compressed_proof_bytes)
}

fn compute_query_set<E: PrimeField>(points: &Vec<E>) -> QuerySet<E> {
    let mut queryset = QuerySet::<E>::new();
    let beta = String::from("beta");
    points.into_iter().enumerate().for_each(|(id, point)| {
        queryset.insert((get_poly_label(id), (beta.clone(), point.clone())));
        println!("Inserting query set {} {} {}", beta, point, get_poly_label(id));
    });

    queryset
}

fn compute_evaluations<E: PrimeField>(log_evals: &Vec<(E, E)>) -> Evaluations<E, E> {
    let mut evalmap = Evaluations::<E, E>::new();
    log_evals.into_iter().enumerate().for_each(|(id, (point, val))| {
        evalmap.insert((get_poly_label(id), point.clone()), val.clone());
        println!("Adding point {} with value {} {}", point, val, get_poly_label(id));
    });

    evalmap
}

fn get_poly_label(id: usize) -> String {
    String::from(format!("masked_poly_{}", id))
}

fn compute_random_commitment_all<E: PrimeField, P: DenseUVPolynomial<E>, PCS: PolynomialCommitment<E, P, PoseidonSponge<E>>>
    (ck: &PCS::CommitterKey, rng: &mut dyn RngCore, opt: &Opt, should_prove: bool)
    -> (Vec<Vec<PCS::Commitment>>, Vec<P>, Vec<PCS::Randomness>) {
    let random_point_vec = parse_randomness_from_log_file(&opt.mpspdz_output_file, opt.party as u64).expect("Unable to parse randomness");
    let mut labeled_polys: Vec<LabeledPolynomial<E, P>> = Vec::new();
    let mut random_polys: Vec<P> = Vec::new();
    for (party_id, random_point) in random_point_vec.iter() {
        assert_eq!(party_id, &(opt.party as u64), "Random point is not for this party, why is it parsed for this party?");
        let random_poly = P::from_coefficients_vec(vec![*random_point]);
        random_polys.push(random_poly.clone());

        let labeled_poly = get_labeled_poly::<E, P>(random_poly, Some("random_poly"));
        labeled_polys.push(labeled_poly);
    }
    if !should_prove {
        labeled_polys = Vec::new();
        random_polys = Vec::new();
    }

    let (comms, rands) = PCS::commit(&ck, &labeled_polys, Some(rng)).unwrap();

    let mut compressed_bytes = Vec::new();
    comms.into_iter().map(|labeled_comm| labeled_comm.commitment().clone()).collect::<Vec<_>>()
        .serialize_compressed(&mut compressed_bytes).unwrap();

    let all_commitments_bytes = MpcMultiNet::broadcast_bytes_unequal(&compressed_bytes);
    let all_commitments = all_commitments_bytes.iter().enumerate().map(|(idx, bytes)| {
        Vec::<PCS::Commitment>::deserialize_compressed(bytes.as_slice()).expect(format!("Unable to deserialize commitment {}", idx).as_str())
    }).collect::<Vec<_>>();

    (all_commitments, random_polys, rands)
}
