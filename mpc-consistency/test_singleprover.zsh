#!/usr/bin/env zsh
set -xe
trap "exit" INT TERM
trap "kill 0" EXIT

cargo build --features parallel --release --bin gen_pp_ped --bin gen_pp_ipa --bin gen_pp_kzg --bin gen_commitments_kzg --bin prove_verify_kzg --bin gen_commitments_ipa --bin prove_verify_ipa --bin gen_commitments_ped --bin prove_verify_ped

BIN=./target/release/gen_commitments_kzg

N_ARGS=6000

./target/release/gen_pp_kzg --num-args $N_ARGS;

$BIN --hosts data/2 --player-input-binary-path "/Users/hidde/PhD/auditing/MP-SPDZ/Player-Data/Input-Binary-P0-0" --party 0 --save & ; pid0=$!
$BIN --hosts data/2 --player-input-binary-path "/Users/hidde/PhD/auditing/MP-SPDZ/Player-Data/Input-Binary-P0-0" --party 1 --save & ; pid1=$!


BIN=./target/release/prove_verify_kzg

$BIN --hosts data/2 --mpspdz-output-file "output-P0.txt" --party 0 --prover-party 0 -d & ; pid0=$!
$BIN --hosts data/2 --mpspdz-output-file "output-P1.txt" --party 1 --prover-party 0 & ; pid1=$!
wait $pid0 $pid1
