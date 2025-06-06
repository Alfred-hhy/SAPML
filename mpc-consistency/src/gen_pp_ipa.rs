
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_ec::AffineRepr;
use ark_ec::bls12::Bls12;
use ark_ec::pairing::Pairing;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use ark_secp256k1::{Affine, Fr};
use blake2::Blake2s256;

mod gen_pp;
mod common;
mod mpspdz;
mod perf_trace_structured;


pub type E = Affine;
pub type UniPoly_377 = DensePolynomial<Fr>;
pub type Sponge_Bls12_377 = PoseidonSponge<<E as AffineRepr>::ScalarField>;


pub type PCS = InnerProductArgPC<E, Blake2s256, UniPoly_377, Sponge_Bls12_377>;

fn main() {
    gen_pp::generate_and_save::<<E as AffineRepr>::ScalarField, UniPoly_377, PCS>();
}
