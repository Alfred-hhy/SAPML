use ark_std::{cfg_into_iter, collections::{BTreeMap, BTreeSet}, fmt::Debug, hash::Hash, iter::FromIterator, string::{String, ToString}, vec::Vec};
use ark_poly_commit::{BatchLCProof, CHALLENGE_SIZE, DenseUVPolynomial, Error, Evaluations, QuerySet};
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, LinearCombination};
use ark_poly_commit::{PCCommitterKey, PCRandomness, PCUniversalParams, PolynomialCommitment};

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use ark_std::{convert::TryInto, end_timer, format, marker::PhantomData, ops::Mul, start_timer, vec};

use rayon::prelude::*;

mod data_structures;
pub use data_structures::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use ark_poly_commit::challenge::ChallengeGenerator;
use ark_crypto_primitives::sponge::CryptographicSponge;
use blake2::digest::Digest;

pub struct BaselinePedPC<
    G: AffineRepr,
    D: Digest,
    P: DenseUVPolynomial<G::ScalarField>,
    S: CryptographicSponge,
> {
    _projective: PhantomData<G>,
    _digest: PhantomData<D>,
    _poly: PhantomData<P>,
    _sponge: PhantomData<S>,
}

impl<G, D, P, S> BaselinePedPC<G, D, P, S>
where
    G: AffineRepr,
    D: Digest,
    P: DenseUVPolynomial<G::ScalarField>,
    S: CryptographicSponge,
{
    pub const PROTOCOL_NAME: &'static [u8] = b"PC-DL-2020-PED";

    fn print_type_of<T>(_: &T) {
        println!("{}", std::any::type_name::<T>())
    }

    fn cm_commit(
        comm_key: G,
        scalars: &[G::ScalarField],
        hiding_generator: G,
        randomizers: &[G::ScalarField],
    ) -> Vec<G::Group> {
        let timerCommit = start_timer!(|| format!("Computing commitment to {} scalars and {} random values!!", scalars.len(), randomizers.len()));

        let iter = cfg_into_iter!(scalars);
        Self::print_type_of(&iter);
        let comm = iter.zip(randomizers).map(|(scalar, rand)| {
            comm_key.into_group().mul(scalar) + hiding_generator.into_group().mul(rand)
        }).collect();


        end_timer!(timerCommit);

        comm
    }

    fn compute_random_oracle_challenge(bytes: &[u8]) -> G::ScalarField {
        let mut i = 0u64;
        let mut challenge = None;
        while challenge.is_none() {
            let mut hash_input = bytes.to_vec();
            hash_input.extend(i.to_le_bytes());
            let hash = D::digest(&hash_input.as_slice());
            challenge = <G::ScalarField as Field>::from_random_bytes(&hash);

            i += 1;
        }

        challenge.unwrap()
    }

    fn sample_generators(num_generators: usize) -> Vec<G> {
        let generators: Vec<_> = cfg_into_iter!(0..num_generators)
            .map(|i| {
                let i = i as u64;
                let mut hash =
                    D::digest([Self::PROTOCOL_NAME, &i.to_le_bytes()].concat().as_slice());
                let mut g = G::from_random_bytes(&hash);
                let mut j = 0u64;
                while g.is_none() {
                    let mut bytes = Self::PROTOCOL_NAME.to_vec();
                    bytes.extend(i.to_le_bytes());
                    bytes.extend(j.to_le_bytes());
                    hash = D::digest(bytes.as_slice());
                    g = G::from_random_bytes(&hash);
                    j += 1;
                }
                let generator = g.unwrap();
                generator.mul_by_cofactor_to_group()
            })
            .collect();

        G::Group::normalize_batch(&generators)
    }
}

impl<G, D, P, S> PolynomialCommitment<G::ScalarField, P, S> for BaselinePedPC<G, D, P, S>
where
    G: AffineRepr,
    G::Group: VariableBaseMSM<MulBase = G>,
    D: Digest,
    P: DenseUVPolynomial<G::ScalarField, Point = G::ScalarField>,
    S: CryptographicSponge,
{
    type UniversalParams = UniversalParams<G>;
    type CommitterKey = CommitterKey<G>;
    type VerifierKey = VerifierKey<G>;
    type PreparedVerifierKey = PreparedVerifierKey<G>;
    type Commitment = Commitment<G>;
    type PreparedCommitment = PreparedCommitment<G>;
    type Randomness = Randomness<G>;
    type Proof = Proof<G>;
    type BatchProof = Vec<Self::Proof>;
    type Error = Error;

    fn setup<R: RngCore>(
        max_degree: usize,
        _: Option<usize>,
        _rng: &mut R,
    ) -> Result<Self::UniversalParams, Self::Error> {
        let max_degree = (max_degree + 1).next_power_of_two() - 1;

        let setup_time = start_timer!(|| format!("Sampling {} generators", 2));
        let mut generators = Self::sample_generators(2);
        end_timer!(setup_time);

        let pp = UniversalParams {
            comm_key: generators[0],
            hiding_comm_key: generators[1]
        };

        Ok(pp)
    }

    fn trim(
        pp: &Self::UniversalParams,
        supported_degree: usize,
        _supported_hiding_bound: usize,
        _enforced_degree_bounds: Option<&[usize]>,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), Self::Error> {
        // }

        let trim_time =
            start_timer!(|| format!("Trimming to supported degree of {}", supported_degree));

        let ck = CommitterKey {
            comm_key: pp.comm_key,
            hiding_comm_key: pp.hiding_comm_key
        };

        let vk = VerifierKey {
            comm_key: pp.comm_key,
            hiding_comm_key: pp.hiding_comm_key
        };

        end_timer!(trim_time);

        Ok((ck, vk))
    }
    fn commit<'a>(
        ck: &Self::CommitterKey,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField, P>>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<
        (
            Vec<LabeledCommitment<Self::Commitment>>,
            Vec<Self::Randomness>,
        ),
        Self::Error,
    >
    where
        P: 'a,
    {
        let rng = &mut ark_poly_commit::optional_rng::OptionalRng(rng);
        let mut comms = Vec::new();
        let mut rands = Vec::new();

        let commit_time = start_timer!(|| "Committing to polynomials");
        for labeled_polynomial in polynomials {

            let polynomial: &P = labeled_polynomial.polynomial();
            let label = labeled_polynomial.label();
            let hiding_bound = labeled_polynomial.degree();
            let degree_bound = labeled_polynomial.degree();

            let commit_time = start_timer!(|| format!(
                "Polynomial {} of degree {}, degree bound {:?}, and hiding bound {:?}",
                label,
                polynomial.degree(),
                degree_bound,
                hiding_bound,
            ));

            let randomness = Randomness::rand(hiding_bound + 1, false, None, rng);

            let comm: Vec<G> = Self::cm_commit(
                ck.comm_key,
                &polynomial.coeffs(),
                ck.hiding_comm_key,
                &randomness.rand,
            )
                .into_iter().map(|x| x.into()).collect();

            let commitment = Commitment { comm };
            let labeled_comm = LabeledCommitment::new(label.to_string(), commitment, None);

            comms.push(labeled_comm);
            rands.push(randomness);

            end_timer!(commit_time);
        }

        end_timer!(commit_time);
        Ok((comms, rands))
    }

    fn open<'a>(
        ck: &Self::CommitterKey,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField, P>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        rands: impl IntoIterator<Item = &'a Self::Randomness>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::Proof, Self::Error>
    where
        Self::Commitment: 'a,
        Self::Randomness: 'a,
        P: 'a,
    {

            let mut polys_iter = labeled_polynomials.into_iter();
            let mut rands_iter = rands.into_iter();
        {
            let combined_polynomial = polys_iter.next().unwrap();
            let combined_rand = rands_iter.next().unwrap();


            let d = combined_polynomial.degree();
            let mut z: Vec<G::ScalarField> = Vec::with_capacity(d + 1);
            let mut cur_z: G::ScalarField = G::ScalarField::one();
            for _ in 0..(d + 1) {
                z.push(cur_z);
                cur_z *= point;
            }
            let mut z = z.as_mut_slice();

            let mut rho_opening = G::ScalarField::zero();
            for i in 0..(d + 1) {
                rho_opening += z[i] * &combined_rand.rand[i];
            }

            Ok(Proof {
                combined_opening: rho_opening
            })
        }
    }

    fn check<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = G::ScalarField>,
        proof: &Self::Proof,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        let check_time = start_timer!(|| "Checking evaluations");

        let mut commitments_iter = commitments.into_iter();
        let mut values_iter = values.into_iter();


        {
            let combined_commitments = commitments_iter.next().unwrap();
            let combined_values = values_iter.next().unwrap();


            let d = combined_commitments.commitment().comm.len();
            println!("commitment size {:?}", d);
            let mut z: Vec<G::ScalarField> = Vec::with_capacity(d + 1);
            let mut cur_z: G::ScalarField = G::ScalarField::one();
            for _ in 0..d {
                z.push(cur_z);
                cur_z *= point;
            }
            let mut z = z.as_mut_slice();


            let scalars_bigint = ark_std::cfg_iter!(z)
                .map(|s| s.into_bigint())
                .collect::<Vec<_>>();

            let rho_com = <G::Group as VariableBaseMSM>::msm_bigint(&combined_commitments.commitment().comm, &scalars_bigint);

            let rho_com_check = vk.comm_key.mul(combined_values) +
                vk.hiding_comm_key.mul(proof.combined_opening);
            let equal = rho_com == rho_com_check;
            end_timer!(check_time);

            Ok(equal)
        }
    }

    fn batch_check<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<P::Point>,
        values: &Evaluations<G::ScalarField, P::Point>,
        proof: &Self::BatchProof,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        panic!("Not implemented!");
    }

    fn open_combinations<'a>(
        ck: &Self::CommitterKey,
        linear_combinations: impl IntoIterator<Item = &'a LinearCombination<G::ScalarField>>,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<G::ScalarField, P>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<P::Point>,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        rands: impl IntoIterator<Item = &'a Self::Randomness>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<BatchLCProof<G::ScalarField, Self::BatchProof>, Self::Error>
    where
        Self::Randomness: 'a,
        Self::Commitment: 'a,
        P: 'a,
    {
        panic!("Not implemented!");

    }

    fn check_combinations<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        linear_combinations: impl IntoIterator<Item = &'a LinearCombination<G::ScalarField>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        eqn_query_set: &QuerySet<P::Point>,
        eqn_evaluations: &Evaluations<P::Point, G::ScalarField>,
        proof: &BatchLCProof<G::ScalarField, Self::BatchProof>,
        opening_challenges: &mut ChallengeGenerator<G::ScalarField, S>,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        panic!("Not implemented!");
    }
}
