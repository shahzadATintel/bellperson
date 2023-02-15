#![allow(clippy::suspicious_arithmetic_impl)]
//! `bellperson` is a crate for building zk-SNARK circuits. It provides circuit
//! traits and and primitive structures, as well as basic gadget implementations
//! such as booleans and number abstractions.
//!
//! # Example circuit
//!
//! Say we want to write a circuit that proves we know the preimage to some hash
//! computed using SHA-256d (calling SHA-256 twice). The preimage must have a
//! fixed length known in advance (because the circuit parameters will depend on
//! it), but can otherwise have any value. We take the following strategy:
//!
//! - Witness each bit of the preimage.
//! - Compute `hash = SHA-256d(preimage)` inside the circuit.
//! - Expose `hash` as a public input using multiscalar packing.
//!
//! ```no_run
//! use bellperson::{
//!     gadgets::{
//!         boolean::{AllocatedBit, Boolean},
//!         multipack,
//!         sha256::sha256,
//!     },
//!     groth16, Circuit, ConstraintSystem, SynthesisError,
//! };
//! use blstrs::Bls12;
//! use ff::PrimeField;
//! use pairing::Engine;
//! use rand::rngs::OsRng;
//! use sha2::{Digest, Sha256};
//!
//! /// Our own SHA-256d gadget. Input and output are in little-endian bit order.
//! fn sha256d<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
//!     mut cs: CS,
//!     data: &[Boolean],
//! ) -> Result<Vec<Boolean>, SynthesisError> {
//!     // Flip endianness of each input byte
//!     let input: Vec<_> = data
//!         .chunks(8)
//!         .map(|c| c.iter().rev())
//!         .flatten()
//!         .cloned()
//!         .collect();
//!
//!     let mid = sha256(cs.namespace(|| "SHA-256(input)"), &input)?;
//!     let res = sha256(cs.namespace(|| "SHA-256(mid)"), &mid)?;
//!
//!     // Flip endianness of each output byte
//!     Ok(res
//!         .chunks(8)
//!         .map(|c| c.iter().rev())
//!         .flatten()
//!         .cloned()
//!         .collect())
//! }
//!
//! struct MyCircuit {
//!     /// The input to SHA-256d we are proving that we know. Set to `None` when we
//!     /// are verifying a proof (and do not have the witness data).
//!     preimage: Option<[u8; 80]>,
//! }
//!
//! impl<Scalar: PrimeField> Circuit<Scalar> for MyCircuit {
//!     fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
//!         // Compute the values for the bits of the preimage. If we are verifying a proof,
//!         // we still need to create the same constraints, so we return an equivalent-size
//!         // Vec of None (indicating that the value of each bit is unknown).
//!         let bit_values = if let Some(preimage) = self.preimage {
//!             preimage
//!                 .iter()
//!                 .map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
//!                 .flatten()
//!                 .map(|b| Some(b))
//!                 .collect()
//!         } else {
//!             vec![None; 80 * 8]
//!         };
//!         assert_eq!(bit_values.len(), 80 * 8);
//!
//!         // Witness the bits of the preimage.
//!         let preimage_bits = bit_values
//!             .into_iter()
//!             .enumerate()
//!             // Allocate each bit.
//!             .map(|(i, b)| {
//!                 AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {}", i)), b)
//!             })
//!             // Convert the AllocatedBits into Booleans (required for the sha256 gadget).
//!             .map(|b| b.map(Boolean::from))
//!             .collect::<Result<Vec<_>, _>>()?;
//!
//!         // Compute hash = SHA-256d(preimage).
//!         let hash = sha256d(cs.namespace(|| "SHA-256d(preimage)"), &preimage_bits)?;
//!
//!         // Expose the vector of 32 boolean variables as compact public inputs.
//!         multipack::pack_into_inputs(cs.namespace(|| "pack hash"), &hash)
//!     }
//! }
//!
//! // Create parameters for our circuit. In a production deployment these would
//! // be generated securely using a multiparty computation.
//! let params = {
//!     let c = MyCircuit { preimage: None };
//!     groth16::generate_random_parameters::<Bls12, _, _>(c, &mut OsRng).unwrap()
//! };
//!
//! // Prepare the verification key (for proof verification).
//! let pvk = groth16::prepare_verifying_key(&params.vk);
//!
//! // Pick a preimage and compute its hash.
//! let preimage = [42; 80];
//! let hash = Sha256::digest(&Sha256::digest(&preimage));
//!
//! // Create an instance of our circuit (with the preimage as a witness).
//! let c = MyCircuit {
//!     preimage: Some(preimage),
//! };
//!
//! // Create a Groth16 proof with our parameters.
//! let proof = groth16::create_random_proof(c, &params, &mut OsRng).unwrap();
//!
//! // Pack the hash as inputs for proof verification.
//! let hash_bits = multipack::bytes_to_bits_le(&hash);
//! let inputs = multipack::compute_multipacking::<<Bls12 as Engine>::Fr>(&hash_bits);
//!
//! // Check the proof!
//! assert!(groth16::verify_proof(&pvk, &proof, &inputs).unwrap());
//! ```
//!
//! # Roadmap
//!
//! `bellperson` is being refactored into a generic proving library. Currently it
//! is pairing-specific, and different types of proving systems need to be
//! implemented as sub-modules. After the refactor, `bellperson` will be generic
//! using the [`ff`] and [`group`] crates, while specific proving systems will
//! be separate crates that pull in the dependencies they require.

#![cfg_attr(all(target_arch = "aarch64", nightly), feature(stdsimd))]

#[cfg(test)]
#[macro_use]
extern crate hex_literal;

pub mod domain;
pub mod gadgets;
pub mod gpu;
#[cfg(feature = "groth16")]
pub mod groth16;
pub mod multiexp;
pub mod util_cs;

mod lc;
pub use lc::{Index, LinearCombination, Variable};
mod constraint_system;
pub use constraint_system::{Circuit, ConstraintSystem, Namespace, SynthesisError};

pub const BELLMAN_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "groth16")]
pub(crate) fn le_bytes_to_u64s(le_bytes: &[u8]) -> Vec<u64> {
    assert_eq!(
        le_bytes.len() % 8,
        0,
        "length must be divisible by u64 byte length (8-bytes)"
    );
    le_bytes
        .chunks(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{fs::File, path::Path};

    use blstrs::{Bls12, Scalar as Fr};
    use ff::Field;
    use log::debug;
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        groth16::{
            create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
            Parameters,
        },
        util_cs::{test_cs::TestConstraintSystem, Comparable},
    };

    #[derive(Clone)]
    struct TestudoCircuit {
        aux: Vec<Fr>,
        inputs: Vec<Fr>,
        num_cons: usize,
    }

    impl TestudoCircuit {
        fn new(num_aux: usize, num_inputs: usize, num_cons: usize) -> Self {
            assert!(num_aux.is_power_of_two());
            assert!(num_cons.is_power_of_two());
            assert!(num_inputs < num_aux);

            let mut rng = StdRng::seed_from_u64(1234);

            let aux = std::iter::repeat_with(|| Fr::random(&mut rng))
                .filter_map(|x| {
                    if bool::from(x.is_zero()) {
                        None
                    } else {
                        Some(x)
                    }
                })
                .take(num_aux)
                .collect();

            let inputs = std::iter::repeat_with(|| Fr::random(&mut rng))
                .filter_map(|x| {
                    if bool::from(x.is_zero()) {
                        None
                    } else {
                        Some(x)
                    }
                })
                .take(num_inputs)
                .collect();

            TestudoCircuit {
                aux,
                inputs,
                num_cons,
            }
        }

        fn pub_inputs(&self) -> &[Fr] {
            &self.inputs
        }
    }

    impl Circuit<Fr> for TestudoCircuit {
        fn synthesize<CS: ConstraintSystem<Fr>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
            // All r1cs variables: `vars = aux || 1 || inputs`.
            let var_values: Vec<Fr> = self
                .aux
                .iter()
                .copied()
                .chain(std::iter::once(Fr::one()))
                .chain(self.inputs.iter().copied())
                .collect();

            // `num_vars = num_aux + 1 + num_inputs`.
            let num_vars = var_values.len();

            // Assign variables.
            let mut vars = Vec::<Variable>::with_capacity(num_vars);
            for (i, aux) in self.aux.into_iter().enumerate() {
                let aux = cs
                    .alloc(|| format!("aux_{}", i), || Ok(aux))
                    .expect("allocation should not fail");
                vars.push(aux);
            }
            vars.push(CS::one());
            for (i, pi) in self.inputs.into_iter().enumerate() {
                let pi = cs
                    .alloc_input(|| format!("pi_{}", i), || Ok(pi))
                    .expect("allocation should not fail");
                vars.push(pi);
            }

            // Add multiplication constraints that are satisfied for the random choice of variable
            // values.
            for i in 0..self.num_cons {
                let a_idx = i % num_vars;
                let b_idx = (i + 2) % num_vars;
                let c_idx = (i + 3) % num_vars;

                let a_var = vars[a_idx];
                let b_var = vars[b_idx];
                let c_var = vars[c_idx];

                let a_val = &var_values[a_idx];
                let b_val = &var_values[b_idx];
                let c_val = &var_values[c_idx];
                let c_coeff = a_val * b_val * c_val.invert().unwrap();

                // (1)a * (1)b = (abc^-1)c
                cs.enforce(
                    || format!("con_{}", i),
                    |lc| lc + a_var,
                    |lc| lc + b_var,
                    |lc| lc + (c_coeff, c_var),
                );
            }

            Ok(())
        }
    }

    #[test]
    fn test_testudo_circ() {
        fil_logger::maybe_init();

        let exp = 4;
        let num_aux = 1 << exp;
        let num_cons = num_aux;
        let num_inputs = 10;
        let params_path = format!("/tmp/bellperson-testudo-{}-{}.params", num_inputs, exp);

        let circ = TestudoCircuit::new(num_aux, num_inputs, num_cons);
        let mut cs = TestConstraintSystem::<Fr>::new();
        debug!("vmx: synthesize circuit: start");
        circ.synthesize(&mut cs).expect("synthesis failed");
        assert!(cs.is_satisfied());
        assert_eq!(cs.aux().len(), num_aux);
        // Add one for `1`.
        assert_eq!(cs.num_inputs(), num_inputs + 1);
        assert_eq!(cs.num_constraints(), num_cons);
        debug!("vmx: synthesize circuit: stop");

        let circ = TestudoCircuit::new(num_aux, num_inputs, num_cons);
        let pub_inputs = circ.pub_inputs().to_vec();
        let mut rng = StdRng::seed_from_u64(1234);
        debug!("vmx: generate parameters: start");
        let params = match Path::new(&params_path).try_exists() {
            Ok(false) => {
                debug!("vmx: parameters file doesn't exist yet, generate parameters and write them to {}", params_path);
                let params: Parameters<Bls12> =
                    generate_random_parameters(circ.clone(), &mut rng).expect("param-gen failed");
                let params_file = File::create(params_path).unwrap();
                params.write(params_file).unwrap();
                params
            }
            _ => {
                debug!("vmx: reading in parameters file from {}", params_path);
                let params_file = File::open(params_path).unwrap();
                Parameters::<Bls12>::read(params_file, true).unwrap()
            }
        };
        debug!("vmx: generate parameters: stop");
        debug!("vmx: generate verifying key: start");
        let pvk = prepare_verifying_key::<Bls12>(&params.vk);
        debug!("vmx: generate verifying key: stop");
        debug!("vmx: start proof: start");
        let proof = create_random_proof(circ, &params, &mut rng).expect("proving failed");
        debug!("vmx: start proof: stop");
        debug!("vmx: verify proof: start");
        assert!(verify_proof(&pvk, &proof, &pub_inputs).expect("verification failed"));
        debug!("vmx: verify proof: stop");
    }
}
