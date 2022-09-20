use bellperson::gadgets::boolean::{AllocatedBit, Boolean};
use bellperson::gadgets::multieq::MultiEq;
use bellperson::gadgets::multipack;
use bellperson::gadgets::multipack::{bytes_to_bits_le, compute_multipacking};
use bellperson::gadgets::uint32::UInt32;
use bellperson::groth16::{
    create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
    Parameters, Proof,
};
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use bitvec::order::Lsb0;
use bitvec::view::BitView;
use blstrs::{Bls12, Scalar};
use ff::PrimeField;
use rand_core::OsRng;
use rc4::{Key, Rc4};
use rc4::{KeyInit, StreamCipher};
use std::convert::TryInto;

const KEY_BIT_LENGTH: usize = 40;
const PLAINTEXT_BIT_LENGTH: usize = 256;

fn rc4_encryption(key: Vec<u8>, plaintext: Vec<u8>) -> Vec<u8> {
    let mut state: [u8; 256] = [0x00; 256];

    // KSA
    for (index, state_element) in state.iter_mut().enumerate() {
        *state_element = index as u8
    }

    let mut j: u32 = 0;

    // mix state bytes
    for state_index in 0..256 {
        j = (j + (state[state_index] as u32) + (key[state_index % key.len()] as u32)) % 256;
        state.swap(state_index, j as usize);
    }

    let mut i: u32 = 0;
    let mut j: u32 = 0;
    let mut result: Vec<u8> = plaintext;

    for result_element in result.iter_mut() {
        // gamma generation
        i = (i + 1) % 256;
        j = (j + state[i as usize] as u32) % 256;
        state.swap(i as usize, j as usize);
        let gamma_index: u32 = (state[i as usize] as u32 + state[j as usize] as u32) % 256;
        let gamma = state[gamma_index as usize];

        // actual encryption in stream ciphers is XORing generated gamma with plaintext
        *result_element ^= gamma;
    }
    result
}

#[test]
fn test_rc4_circuit() {
    fn positive_test(
        key: [u8; KEY_BIT_LENGTH / 8],
        plaintext: [u8; PLAINTEXT_BIT_LENGTH / 8],
        use_verifier_circuit_to_generate_params: bool,
    ) {
        println!("positive test");

        // reference RC4 implementation
        let mut rc4 = Rc4::new(&Key::from(key));
        let mut expected_ciphertext = plaintext; // initially ciphertext buffer should contain plaintext
        rc4.apply_keystream(&mut expected_ciphertext); // encrypts ciphertext buffer in place

        // self-made RC4 implementation
        let ciphertext = rc4_encryption(key.to_vec(), plaintext.to_vec());
        assert_eq!(expected_ciphertext.to_vec(), ciphertext);

        // Prove that I know secret combination of key and plaintext that will correspond to the given ciphertext (RC4)
        let mut rng = OsRng;
        let circuit_verifier = RC4Circuit::circuit_verifier();
        let circuit_prover = RC4Circuit::circuit_prover(key.to_vec(), plaintext.to_vec());
        let public_inputs_for_verification =
            circuit_verifier.compute_public_inputs(ciphertext.to_vec());
        for scalar in public_inputs_for_verification.clone() {
            println!("expected num: {:?}", scalar);
        }
        // Generate global parameters common for both prover and verifier
        let parameters: Parameters<Bls12> = if use_verifier_circuit_to_generate_params {
            generate_random_parameters(circuit_verifier, &mut rng)
                .expect("can't generate global parameters")
        } else {
            generate_random_parameters(circuit_prover.clone(), &mut rng)
                .expect("can't generate global parameters")
        };
        let verifier_key = prepare_verifying_key(&parameters.vk);

        // Generate proof
        let proof = create_random_proof(circuit_prover, &parameters, &mut rng)
            .expect("can't generate proof");
        let mut proof_bytes = vec![];
        proof
            .write(&mut proof_bytes)
            .expect("can't serialize proof");

        // Verify proof
        let proof = Proof::<Bls12>::read(&proof_bytes[..]).expect("can't deserialize proof");
        assert!(
            verify_proof(&verifier_key, &proof, &public_inputs_for_verification)
                .expect("can't verify")
        );
        println!("##################");
    }

    fn negative_test(
        key: [u8; KEY_BIT_LENGTH / 8],
        plaintext: [u8; PLAINTEXT_BIT_LENGTH / 8],
        ciphertext: [u8; PLAINTEXT_BIT_LENGTH / 8],
        use_verifier_circuit_to_generate_params: bool,
    ) {
        println!("negative test");

        // Prove that I know secret combination of key and plaintext that will correspond to the given ciphertext (RC4)
        let mut rng = OsRng;
        let circuit_verifier = RC4Circuit::circuit_verifier();
        let circuit_prover = RC4Circuit::circuit_prover(key.to_vec(), plaintext.to_vec());
        let public_inputs_for_verification =
            circuit_verifier.compute_public_inputs(ciphertext.to_vec());
        for scalar in public_inputs_for_verification.clone() {
            println!("expected num: {:?}", scalar);
        }
        // Generate global parameters common for both prover and verifier
        let parameters: Parameters<Bls12> = if use_verifier_circuit_to_generate_params {
            generate_random_parameters(circuit_verifier, &mut rng)
                .expect("can't generate global parameters")
        } else {
            generate_random_parameters(circuit_prover.clone(), &mut rng)
                .expect("can't generate global parameters")
        };
        let verifier_key = prepare_verifying_key(&parameters.vk);

        // Generate proof
        let proof = create_random_proof(circuit_prover, &parameters, &mut rng)
            .expect("can't generate proof");
        let mut proof_bytes = vec![];
        proof
            .write(&mut proof_bytes)
            .expect("can't serialize proof");

        // Verify proof
        let proof = Proof::<Bls12>::read(&proof_bytes[..]).expect("can't deserialize proof");
        assert!(
            !verify_proof(&verifier_key, &proof, &public_inputs_for_verification)
                .expect("can't verify")
        );
        println!("##################");
    }

    // Successful test 1
    let key = [0x00, 0x00, 0x00, 0x00, 0x00];
    let plaintext = [0x00; 32];
    positive_test(key, plaintext, false);

    // Successful test 2
    let key = [0x00, 0x00, 0x00, 0x00, 0x00];
    let plaintext = [0xcd; 32]; // with arbitrary plaintext
    positive_test(key, plaintext, false);

    // Successful test 3
    let key = [0x00, 0x00, 0x00, 0x00, 0x00];
    let plaintext = [0x00; 32];
    let ciphertext = [0x01; 32]; // with something arbitrary (not RC4 encrypted)
    negative_test(key, plaintext, ciphertext, false);

    // Successful test 4
    let key = [0x00, 0x00, 0x00, 0x00, 0x00];
    let plaintext = [0x00; 32];
    let ciphertext = [0x01; 32]; // with something arbitrary (not RC4 encrypted)
    negative_test(key, plaintext, ciphertext, true);

    // Successful test 5
    let key = [0x00, 0xfa, 0x00, 0x00, 0x01]; // with arbitrary key
    let plaintext = [0xfe; 32]; // with arbitrary plaintext
    positive_test(key, plaintext, false);

    // Failed test 6
    // Don't know why currently, but if we use verifier's circuit where key/plaintext are arbitrary and their values are undefined
    // (when we use verifier's circuit for generating global parameters), we have error while verification
    //let key = [0x00, 0xfa, 0x00, 0x00, 0x01]; // with arbitrary key
    //let plaintext = [0xfe; 32]; // with arbitrary plaintext
    //positive_test(key, plaintext, true);
}

#[derive(Clone, Debug)]
struct RC4Circuit {
    key: Option<[bool; KEY_BIT_LENGTH]>,
    plaintext: Option<[bool; PLAINTEXT_BIT_LENGTH]>,
}

impl RC4Circuit {
    fn circuit_prover(key: Vec<u8>, plaintext: Vec<u8>) -> Self {
        // Specify input of computation (convert from bytes to bits as it is required by circuits)
        let key = Some(bytes_to_bits_le(key.as_slice()).try_into().unwrap_or_else(
            |v: Vec<bool>| {
                panic!(
                    "Expected a Vec of length {} but it was {}",
                    KEY_BIT_LENGTH / 8,
                    v.len()
                )
            },
        ));
        let plaintext = Some(
            bytes_to_bits_le(plaintext.as_slice())
                .try_into()
                .unwrap_or_else(|v: Vec<bool>| {
                    panic!(
                        "Expected a Vec of length {} but it was {}",
                        PLAINTEXT_BIT_LENGTH / 8,
                        v.len()
                    )
                }),
        );

        RC4Circuit { key, plaintext }
    }

    fn circuit_verifier() -> Self {
        RC4Circuit {
            key: None,
            plaintext: None,
        }
    }

    fn compute_public_inputs(&self, ciphertext: Vec<u8>) -> Vec<Scalar> {
        // Specify output of computation (required for both prover and verifier)
        let public_inputs_verification =
            compute_multipacking(bytes_to_bits_le(ciphertext.as_slice()).as_slice());
        public_inputs_verification
    }
}

// Synthesize circuit that enforces RC4 encryption operation
impl<Scalar: PrimeField> Circuit<Scalar> for RC4Circuit {
    fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        fn uint32_to_usize(value: UInt32) -> usize {
            boolean_slice_to_usize(value.into_bits().as_slice())
        }

        fn boolean_slice_to_usize(slice: &[Boolean]) -> usize {
            let mut result = 0usize;
            for (i, bit) in slice.iter().enumerate() {
                if let Some(true) = bit.get_value() {
                    result += 1 << i;
                }
            }
            result
        }

        let mut cs = MultiEq::new(cs);

        let key_bits: Vec<Option<bool>> = if self.key.is_none() {
            vec![None; KEY_BIT_LENGTH]
        } else {
            self.key.unwrap().to_vec().into_iter().map(Some).collect()
        };

        let plaintext_bits: Vec<Option<bool>> = if self.plaintext.is_none() {
            vec![None; PLAINTEXT_BIT_LENGTH]
        } else {
            self.plaintext
                .unwrap()
                .to_vec()
                .into_iter()
                .map(Some)
                .collect()
        };

        let key_bits_constrained: Vec<Boolean> = key_bits
            .into_iter()
            .enumerate()
            .map(|(index, key_bit)| {
                AllocatedBit::alloc(cs.namespace(|| format!("key bit {}", index)), key_bit)
                    .map(Into::into)
            })
            .collect::<Result<Vec<Boolean>, SynthesisError>>()?;

        let plaintext_bits_constrained: Vec<Boolean> = plaintext_bits
            .into_iter()
            .enumerate()
            .map(|(index, plaintext_bit)| {
                AllocatedBit::alloc(
                    cs.namespace(|| format!("plaintext bit {}", index)),
                    plaintext_bit,
                )
                .map(Into::into)
            })
            .collect::<Result<Vec<Boolean>, SynthesisError>>()?;

        // KSA
        let mut state: [u8; 256] = [0x00; 256];
        for (index, state_element) in state.iter_mut().enumerate() {
            *state_element = index as u8
        }

        let mut state_bits_constrained = (0..256)
            .map(|byte_index| {
                (0..8)
                    .map(|bit_index| {
                        Boolean::from(
                            AllocatedBit::alloc(
                                cs.namespace(|| {
                                    format!("state bit {}", byte_index * 8 + bit_index)
                                }),
                                Some(
                                    *state[byte_index]
                                        .view_bits::<Lsb0>()
                                        .get(bit_index)
                                        .as_deref()
                                        .unwrap(),
                                ),
                            )
                            // safe, since state is always initialized by constants
                            .unwrap(),
                        )
                    })
                    .collect()
            })
            .collect::<Vec<Vec<Boolean>>>()
            .into_iter()
            .flatten()
            .collect::<Vec<Boolean>>();

        let zeroes32bits = vec![Boolean::constant(false); 32];
        let mut computed_index = 0;
        for state_index in 0..256 {
            let state_slice = &state_bits_constrained[state_index * 8..state_index * 8 + 8];
            let key_slice = &key_bits_constrained[state_index.rem_euclid(KEY_BIT_LENGTH / 8) * 8
                ..state_index.rem_euclid(KEY_BIT_LENGTH / 8) * 8 + 8];
            {
                let j = UInt32::constant(computed_index as u32);

                let a = UInt32::addmany(
                    cs.namespace(|| format!("addition_a {}", state_index)),
                    &[
                        j,
                        UInt32::from_bits(&[state_slice, &zeroes32bits[0..24]].concat()),
                    ],
                )
                .unwrap();
                let key_byte = UInt32::from_bits(&[key_slice, &zeroes32bits[0..24]].concat());
                let b = UInt32::addmany(
                    cs.namespace(|| format!("addition_b {}", state_index)),
                    &[a, key_byte],
                )
                .unwrap();

                let b = &mut b.into_bits()[0..8];
                b.reverse();

                let j = UInt32::from_bits_be(&[&zeroes32bits[0..24], b].concat());
                computed_index = uint32_to_usize(j);
            };

            state_bits_constrained.swap(state_index * 8, computed_index * 8);
            state_bits_constrained.swap(state_index * 8 + 1, computed_index * 8 + 1);
            state_bits_constrained.swap(state_index * 8 + 2, computed_index * 8 + 2);
            state_bits_constrained.swap(state_index * 8 + 3, computed_index * 8 + 3);
            state_bits_constrained.swap(state_index * 8 + 4, computed_index * 8 + 4);
            state_bits_constrained.swap(state_index * 8 + 5, computed_index * 8 + 5);
            state_bits_constrained.swap(state_index * 8 + 6, computed_index * 8 + 6);
            state_bits_constrained.swap(state_index * 8 + 7, computed_index * 8 + 7);
        }

        let mut xor_result: Vec<Vec<Boolean>> = vec![];
        let mut i_index: usize = 0;
        let mut j_index: usize = 0;

        let zeroes32bits = vec![Boolean::constant(false); 32];
        for byte_index in 0..PLAINTEXT_BIT_LENGTH / 8 {
            let i: UInt32 = {
                i_index = (i_index + 1).rem_euclid(256);
                UInt32::constant(i_index as u32)
            };
            let i_usize = uint32_to_usize(i.clone());
            let j: UInt32 = {
                let j = UInt32::constant(j_index as u32);
                let state_slice = &mut state_bits_constrained[i_usize * 8..i_usize * 8 + 8];

                let j = UInt32::addmany(
                    cs.namespace(|| format!("addition_j_index {}", byte_index)),
                    &[
                        j,
                        UInt32::from_bits(&[state_slice, &zeroes32bits[0..24]].concat()),
                    ],
                )
                .unwrap();
                let j = &mut j.into_bits()[0..8];
                j.reverse();
                let j = UInt32::from_bits_be(&[&zeroes32bits[0..24], j].concat());
                j_index = uint32_to_usize(j.clone());
                j
            };
            let j_usize = uint32_to_usize(j.clone());
            state_bits_constrained.swap(i_usize * 8, j_usize * 8);
            state_bits_constrained.swap(i_usize * 8 + 1, j_usize * 8 + 1);
            state_bits_constrained.swap(i_usize * 8 + 2, j_usize * 8 + 2);
            state_bits_constrained.swap(i_usize * 8 + 3, j_usize * 8 + 3);
            state_bits_constrained.swap(i_usize * 8 + 4, j_usize * 8 + 4);
            state_bits_constrained.swap(i_usize * 8 + 5, j_usize * 8 + 5);
            state_bits_constrained.swap(i_usize * 8 + 6, j_usize * 8 + 6);
            state_bits_constrained.swap(i_usize * 8 + 7, j_usize * 8 + 7);

            let gamma_index: usize = {
                let mut state_j = state_bits_constrained.clone();

                let state_i = &mut state_bits_constrained[i_usize * 8..i_usize * 8 + 8];

                let state_j = &mut state_j[j_usize * 8..j_usize * 8 + 8];

                let gamma_index = UInt32::addmany(
                    cs.namespace(|| format!("addition_gamma {}", byte_index)),
                    &[
                        UInt32::from_bits(&[state_i, &zeroes32bits[0..24]].concat()),
                        UInt32::from_bits(&[state_j, &zeroes32bits[0..24]].concat()),
                    ],
                )
                .unwrap();
                let gamma_index = &mut gamma_index.into_bits()[0..8];
                gamma_index.reverse();
                let gamma_index =
                    UInt32::from_bits_be(&[&zeroes32bits[0..24], gamma_index].concat());
                uint32_to_usize(gamma_index)
            };

            let single_byte_xor_result: Vec<Boolean> = (0..8)
                .map(|bit_index| {
                    Boolean::xor(
                        cs.namespace(|| format!("xor constraint {}", byte_index * 8 + bit_index)),
                        plaintext_bits_constrained
                            .get(byte_index * 8 + bit_index)
                            .unwrap(),
                        state_bits_constrained
                            .get(gamma_index * 8 + bit_index)
                            .unwrap(),
                    )
                    .expect("xor panic")
                })
                .collect();
            xor_result.push(single_byte_xor_result);
        }
        let ciphertext = xor_result.into_iter().flatten().collect::<Vec<Boolean>>();

        // Expose final result of computations to CS
        multipack::pack_into_inputs(cs.namespace(|| "pack result"), &ciphertext)
            .expect("can't pack into inputs");

        Ok(())
    }
}
