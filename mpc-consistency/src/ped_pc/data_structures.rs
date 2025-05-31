use ark_poly_commit::*;
use ark_poly_commit::{PCCommitterKey, PCVerifierKey};
use ark_ec::AffineRepr;
use ark_ff::{Field, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;
use ark_std::vec;
use derivative::Derivative;

#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(Default(bound = ""), Clone(bound = ""), Debug(bound = ""))]
pub struct UniversalParams<G: AffineRepr> {
    pub comm_key: G,
    pub hiding_comm_key: G,
}

impl<G: AffineRepr> PCUniversalParams for UniversalParams<G> {
    fn max_degree(&self) -> usize {
        1
    }
}

#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = "")
)]
pub struct CommitterKey<G: AffineRepr> {
    pub comm_key: G,
    pub hiding_comm_key: G

}

impl<G: AffineRepr> PCCommitterKey for CommitterKey<G> {
    fn max_degree(&self) -> usize {
        1
    }
    fn supported_degree(&self) -> usize {
        1
    }
}

pub type VerifierKey<G> = CommitterKey<G>;

impl<G: AffineRepr> PCVerifierKey for VerifierKey<G> {
    fn max_degree(&self) -> usize {
        1
    }

    fn supported_degree(&self) -> usize {
        1
    }
}

pub type PreparedVerifierKey<G> = VerifierKey<G>;

impl<G: AffineRepr> PCPreparedVerifierKey<VerifierKey<G>> for PreparedVerifierKey<G> {
    fn prepare(vk: &VerifierKey<G>) -> Self {
        vk.clone()
    }
}

#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct Commitment<G: AffineRepr> {
    pub comm: Vec<G>,
}

impl<G: AffineRepr> PCCommitment for Commitment<G> {
    #[inline]
    fn empty() -> Self {
        Commitment {
            comm: Vec::new()
        }
    }

    fn has_degree_bound(&self) -> bool {
        false
    }
}


pub type PreparedCommitment<E> = Commitment<E>;

impl<G: AffineRepr> PCPreparedCommitment<Commitment<G>> for PreparedCommitment<G> {
    fn prepare(vk: &Commitment<G>) -> Self {
        vk.clone()
    }
}


#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct Randomness<G: AffineRepr> {
    pub rand: Vec<G::ScalarField>,
}

impl<G: AffineRepr> PCRandomness for Randomness<G> {
    fn empty() -> Self {
        Self {
            rand: Vec::new(),
        }
    }

    fn rand<R: RngCore>(h: usize, has_degree_bound: bool, _: Option<usize>, rng: &mut R) -> Self {
        let rand = (0..h).map(|_| G::ScalarField::rand(rng)).collect();

        Self { rand }
    }
}

#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Debug(bound = "")
)]
pub struct Proof<G: AffineRepr> {
    pub combined_opening: G::ScalarField
}