use bellperson::gadgets::boolean::{AllocatedBit, Boolean};
use bellperson::gadgets::multieq::MultiEq;
use bellperson::gadgets::multipack::{
    bytes_to_bits, bytes_to_bits_le, compute_multipacking, pack_into_inputs,
};
use bellperson::gadgets::uint32::UInt32;
use bellperson::groth16::{
    create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
    Parameters,
};
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use blstrs::{Bls12, Scalar};
use digest::Digest;
use ff::PrimeField;
use itertools::Itertools;
use rand_core::OsRng;
use sha2::Sha256;
use std::convert::TryInto;

const PREIMAGE_LENGTH_BITS: usize = 1024;
const SHA256_SINGLE_BLOCK_SIZE_BITS: usize = 512;
const SHA256_HASH_LENGTH_BITS: usize = 256;

#[allow(clippy::unreadable_literal)]
const ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[allow(clippy::unreadable_literal)]
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

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

fn sha256_process_single_block(block: &[Boolean], current_hash: [u32; 8]) -> [u32; 8] {
    assert_eq!(block.len(), 512);

    let mut w = vec![UInt32::constant(0); 64];
    for index in 0..16 {
        w[index] = UInt32::from_bits_be(&block[index * 32..index * 32 + 32]);
    }

    for index in 16..64 {
        //s0 := (w[i-15] rightrotate  7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift  3)
        let s0 = w[index - 15]
            .rotr(7)
            .xor_no_cs(w[index - 15].rotr(18))
            .xor_no_cs(w[index - 15].shr(3));
        //s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift 10)
        let s1 = w[index - 2]
            .rotr(17)
            .xor_no_cs(w[index - 2].rotr(19))
            .xor_no_cs(w[index - 2].shr(10));
        //w[i] := w[i-16] + s0 + w[i-7] + s1
        let sum = uint32_to_usize(w[index - 16].clone())
            + uint32_to_usize(s0)
            + uint32_to_usize(w[index - 7].clone())
            + uint32_to_usize(s1);
        w[index] = UInt32::constant(sum as u32);
    }

    let mut a = UInt32::constant(current_hash[0]);
    let mut b = UInt32::constant(current_hash[1]);
    let mut c = UInt32::constant(current_hash[2]);
    let mut d = UInt32::constant(current_hash[3]);
    let mut e = UInt32::constant(current_hash[4]);
    let mut f = UInt32::constant(current_hash[5]);
    let mut g = UInt32::constant(current_hash[6]);
    let mut h = current_hash[7] as usize;

    for index in 0..64 {
        //S1 := (e rightrotate 6) xor (e rightrotate 11) xor (e rightrotate 25)
        let s1 = e.rotr(6).xor_no_cs(e.rotr(11)).xor_no_cs(e.rotr(25));

        //ch := (e and f) xor ((not e) and g)
        let ch = e
            .and_no_cs(f.clone())
            .xor_no_cs(e.not_no_cs().and_no_cs(g.clone()));

        //temp1 := h + S1 + ch + k[i] + w[i]
        let temp1 = h
            + uint32_to_usize(s1)
            + uint32_to_usize(ch)
            + (ROUND_CONSTANTS[index] as usize)
            + uint32_to_usize(w[index].clone());

        //S0 := (a rightrotate 2) xor (a rightrotate 13) xor (a rightrotate 22)
        let s0 = a.rotr(2).xor_no_cs(a.rotr(13).xor_no_cs(a.rotr(22)));

        //maj := (a and b) xor (a and c) xor (b and c)
        let maj = a
            .and_no_cs(b.clone())
            .xor_no_cs(a.and_no_cs(c.clone()).xor_no_cs(b.and_no_cs(c.clone())));

        //temp2 := S0 + maj
        let temp2 = uint32_to_usize(s0) + uint32_to_usize(maj);

        h = uint32_to_usize(g.clone());
        g = f.clone();
        f = e.clone();
        e = UInt32::constant((uint32_to_usize(d.clone()) + temp1) as u32);
        d = c.clone();
        c = b.clone();
        b = a.clone();
        a = UInt32::constant((temp1 + temp2) as u32);
    }

    let h0 = (current_hash[0] as usize + uint32_to_usize(a.clone())) as u32;
    let h1 = (current_hash[1] as usize + uint32_to_usize(b.clone())) as u32;
    let h2 = (current_hash[2] as usize + uint32_to_usize(c.clone())) as u32;
    let h3 = (current_hash[3] as usize + uint32_to_usize(d.clone())) as u32;
    let h4 = (current_hash[4] as usize + uint32_to_usize(e.clone())) as u32;
    let h5 = (current_hash[5] as usize + uint32_to_usize(f.clone())) as u32;
    let h6 = (current_hash[6] as usize + uint32_to_usize(g.clone())) as u32;
    let h7 = (current_hash[7] as usize + h) as u32;

    [h0, h1, h2, h3, h4, h5, h6, h7]
}

fn sha256_hashing(message: [u8; PREIMAGE_LENGTH_BITS / 8]) -> [u8; SHA256_HASH_LENGTH_BITS / 8] {
    assert_eq!(8 * message.len() % 512, 0);

    /*
     Pre-processing (Padding):
        begin with the original message of length L bits
        append a single '1' bit
        append K '0' bits, where K is the minimum number >= 0 such that (L + 1 + K + 64) is a multiple of 512
        append L as a 64-bit big-endian integer, making the total post-processed length a multiple of 512 bits
        such that the bits in the message are: <original message of length L> 1 <K zeros> <L as 64 bit integer> , (the number of bits will be a multiple of 512)
    */

    let message_bits = bytes_to_bits(message.to_vec().as_slice())
        .into_iter()
        .map(|bit| Boolean::constant(bit))
        .collect::<Vec<Boolean>>();

    let l = message.len() * 8;
    let mut k = 0;
    while (l + 1 + k + 64) % 512 != 0 {
        k = k + 1
    }

    let k_zeroes = vec![Boolean::constant(false); k];

    let l_bits = bytes_to_bits((l as u64).to_be_bytes().to_vec().as_slice())
        .into_iter()
        .map(|bit| Boolean::constant(bit))
        .collect::<Vec<Boolean>>();

    let padded_message = [
        message_bits,
        vec![Boolean::constant(true); 1],
        k_zeroes,
        l_bits,
    ]
    .concat();

    // Processing
    let mut curr_hash = IV;
    for chunk in padded_message.chunks(512) {
        curr_hash = sha256_process_single_block(chunk, curr_hash);
    }

    let result = curr_hash
        .iter()
        .flat_map(|word| word.clone().to_be_bytes().to_vec())
        .collect::<Vec<u8>>();

    // we know definitely that result is actually a 8 32-bit words
    result.try_into().unwrap()
}

#[derive(Default, Clone, Debug)]
struct Sha256Circuit {
    preimage: Option<[u8; PREIMAGE_LENGTH_BITS / 8]>,
}

fn uint32_into_boolean_vec_le<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
    mut cs: CS,
    value: Option<UInt32>,
) -> Result<Vec<Boolean>, SynthesisError> {
    let values = match value {
        Some(value) => {
            let mut tmp = Vec::with_capacity(32);
            for bit in value.into_bits() {
                tmp.push(bit.get_value());
            }

            tmp
        }
        None => vec![None; 32],
    };

    let bits = values
        .into_iter()
        .enumerate()
        .map(|(i, b)| {
            Ok(Boolean::from(AllocatedBit::alloc(
                cs.namespace(|| format!("bit {}", i)),
                b,
            )?))
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    Ok(bits)
}

fn sha256_process_block<Scalar, CS>(
    mut cs: CS,
    block: &[Option<bool>],
    current_hash: Vec<UInt32>,
) -> Result<Vec<UInt32>, SynthesisError>
where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
{
    assert_eq!(current_hash.len() * 32, SHA256_HASH_LENGTH_BITS);
    assert_eq!(block.len(), SHA256_SINGLE_BLOCK_SIZE_BITS);

    let mut cs = MultiEq::new(&mut cs);

    // if block is undefined, we consider that it is all zeroes, otherwise we are not able to synthesize circuit
    let block = block
        .into_iter()
        .map(|bit| {
            let bit = match bit {
                None => false,
                Some(bit) => *bit,
            };
            bit
        })
        .collect::<Vec<bool>>();

    let block_bits_constrained = block
        .into_iter()
        .enumerate()
        .map(|(index, preimage_bit)| {
            AllocatedBit::alloc(
                cs.namespace(|| format!("preimage bit {}", index)),
                Some(preimage_bit),
            )
            .map(Into::into)
        })
        .collect::<Result<Vec<Boolean>, SynthesisError>>()?;

    let current_hash_constrained: Vec<Boolean> = current_hash
        .iter()
        .enumerate()
        .map(|(word_index, preimage_word)| {
            uint32_into_boolean_vec_le(
                cs.namespace(|| format!("preimage_word {}", word_index)),
                Some(preimage_word.clone()),
            )
        })
        .flatten_ok()
        .collect::<Result<Vec<Boolean>, SynthesisError>>()?;

    let mut w = vec![UInt32::constant(0); 64];
    for index in 0..16 {
        w[index] = UInt32::from_bits_be(&block_bits_constrained[index * 32..index * 32 + 32]);
    }

    for index in 16..64 {
        // s0 := (w[i-15] rightrotate  7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift  3)
        let s0 = w[index - 15]
            .rotr(7)
            .xor(
                cs.namespace(|| format!("loop 1 s0 xor 1 index {}", index)),
                &w[index - 15].rotr(18),
            )?
            .xor(
                cs.namespace(|| format!("loop 1 s0 xor 2 index {}", index)),
                &w[index - 15].shr(3),
            )?;

        // s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift 10)
        let s1 = w[index - 2]
            .rotr(17)
            .xor(
                cs.namespace(|| format!("loop 1 s1 xor 1 index {}", index)),
                &w[index - 2].rotr(19),
            )?
            .xor(
                cs.namespace(|| format!("loop 1 s1 xor 2 index {}", index)),
                &w[index - 2].shr(10),
            )?;

        // w[i] := w[i-16] + s0 + w[i-7] + s1
        let sum = UInt32::addmany(
            cs.namespace(|| format!("w[{}] add many", index)),
            &[w[index - 16].clone(), s0, w[index - 7].clone(), s1],
        )?;
        w[index] = sum
    }

    let mut a = current_hash[0].clone();
    let mut b = current_hash[1].clone();
    let mut c = current_hash[2].clone();
    let mut d = current_hash[3].clone();
    let mut e = current_hash[4].clone();
    let mut f = current_hash[5].clone();
    let mut g = current_hash[6].clone();
    let mut h = current_hash[7].clone();

    for index in 0..64 {
        // s1 := (e rightrotate 6) xor (e rightrotate 11) xor (e rightrotate 25)
        let s1 = e
            .rotr(6)
            .xor(
                cs.namespace(|| format!("loop 2 s1 xor 1 index {}", index)),
                &e.rotr(11),
            )?
            .xor(
                cs.namespace(|| format!("loop 2 s1 xor 2 index {}", index)),
                &e.rotr(25),
            )?;

        // this can be replaced with helper functions (triop / sha256_maj / sha256_ch), but let's do all steps using low-level gadgets
        let e_and_f = e
            .clone()
            .and(cs.namespace(|| format!("e and f {}", index)), &f)?;

        let g_allocated_bits: Vec<AllocatedBit> = g
            .clone()
            .into_bits()
            .into_iter()
            .enumerate()
            .map(|(bit_index, x)| {
                AllocatedBit::alloc(
                    cs.namespace(|| format!("allocate g bit {} for {}", bit_index, index)),
                    x.get_value(),
                )
            })
            .collect::<Result<Vec<AllocatedBit>, SynthesisError>>()?;

        let e_allocated_bits: Vec<AllocatedBit> = e
            .clone()
            .into_bits()
            .into_iter()
            .enumerate()
            .map(|(bit_index, x)| {
                AllocatedBit::alloc(
                    cs.namespace(|| format!("allocate e bit {} for {}", bit_index, index)),
                    x.get_value(),
                )
            })
            .collect::<Result<Vec<AllocatedBit>, SynthesisError>>()?;

        let g_and_not_e = g_allocated_bits
            .into_iter()
            .zip(e_allocated_bits.into_iter())
            .enumerate()
            .map(|(bit_index, bits_g_e)| {
                AllocatedBit::and_not(
                    cs.namespace(|| format!("g and not_e bit {} for {}", bit_index, index)),
                    &bits_g_e.0,
                    &bits_g_e.1,
                )
                .map(Boolean::from)
            })
            .collect::<Result<Vec<Boolean>, SynthesisError>>()?;

        let g_and_not_e = UInt32::from_bits(&g_and_not_e[..]);
        let ch = e_and_f.xor(cs.namespace(|| format!("ch xor {}", index)), &g_and_not_e)?;

        // with deferring temp1 computation we can save some constraints as we call UInt32::add_many only once
        let temp1 = vec![
            h,
            s1,
            ch,
            UInt32::constant(ROUND_CONSTANTS[index]),
            w[index].clone(),
        ];
        let s0 = a
            .rotr(2)
            .xor(cs.namespace(|| format!("s0 xor 1 {}", index)), &a.rotr(13))?
            .xor(cs.namespace(|| format!("s0 xor 2 {}", index)), &a.rotr(22))?;

        let a_and_b = a.and(cs.namespace(|| format!("maj a and b {}", index)), &b)?;
        let a_and_c = a.and(cs.namespace(|| format!("maj a and c {}", index)), &c)?;
        let b_and_c = b.and(cs.namespace(|| format!("maj b and c {}", index)), &c)?;

        let maj = a_and_b
            .xor(cs.namespace(|| format!("maj xor 1 {}", index)), &a_and_c)?
            .xor(cs.namespace(|| format!("maj xor 2 {}", index)), &b_and_c)?;

        let temp2 = vec![s0, maj];

        h = g.clone();
        g = f.clone();
        f = e.clone();
        e = UInt32::addmany(
            cs.namespace(|| format!("e {}", index)),
            &[vec![d], temp1.clone()].concat(),
        )?;
        d = c.clone();
        c = b.clone();
        b = a.clone();
        a = UInt32::addmany(
            cs.namespace(|| format!("a {}", index)),
            &[temp1, temp2].concat(),
        )?;
    }

    let block_hash = current_hash_constrained
        .chunks(32)
        .into_iter()
        .zip(
            [
                a.clone(),
                b.clone(),
                c.clone(),
                d.clone(),
                e.clone(),
                f.clone(),
                g.clone(),
                h.clone(),
            ]
            .iter(),
        )
        .enumerate()
        .map(|(index, (chunk, word))| {
            UInt32::addmany(
                cs.namespace(|| format!("current block {} hashing result", index)),
                &[UInt32::from_bits(chunk), word.clone()],
            )
        })
        .collect::<Result<Vec<UInt32>, SynthesisError>>()?;

    Ok(block_hash)
}

impl Sha256Circuit {
    fn circuit_prover(preimage: Vec<u8>) -> Self {
        let preimage = Some(preimage.try_into().unwrap_or_else(|v: Vec<u8>| {
            panic!(
                "Expected a Vec of length {} but it was {}",
                PREIMAGE_LENGTH_BITS / 8,
                v.len()
            )
        }));
        Sha256Circuit { preimage }
    }

    fn circuit_verifier() -> Self {
        Sha256Circuit { preimage: None }
    }
    fn compute_public_inputs(&self, hash: Vec<u8>) -> Vec<Scalar> {
        compute_multipacking::<Scalar>(bytes_to_bits_le(hash.as_slice()).as_slice())
    }
}

impl<Scalar: PrimeField> Circuit<Scalar> for Sha256Circuit {
    fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let preimage_bits: Vec<Option<bool>> = match self.preimage {
            Some(preimage) => bytes_to_bits(preimage.to_vec().as_slice())
                .into_iter()
                .map(|bit| Some(bit))
                .collect::<Vec<Option<bool>>>(),
            None => vec![None; PREIMAGE_LENGTH_BITS],
        };

        // as our circuit is bounded we can use only fixed-size preimages for proving
        assert_eq!(preimage_bits.len(), PREIMAGE_LENGTH_BITS);

        let l = preimage_bits.len();
        let mut k = 0;
        while (l + 1 + k + 64) % 512 != 0 {
            k = k + 1
        }

        let k_zeroes = vec![Some(false); k];

        let l_bits = bytes_to_bits((l as u64).to_be_bytes().to_vec().as_slice())
            .into_iter()
            .map(|bit| Some(bit))
            .collect::<Vec<Option<bool>>>();

        let padded_message = [preimage_bits, vec![Some(true); 1], k_zeroes, l_bits].concat();

        // Processing
        let mut cur_hash = IV.iter().map(|x| UInt32::constant(x.clone())).collect();
        for (index, chunk) in padded_message.chunks(512).enumerate() {
            cur_hash = sha256_process_block(
                cs.namespace(|| format!("sha256 block processing {}", index)),
                chunk,
                cur_hash,
            )?;
        }

        let output = cur_hash
            .into_iter()
            .rev()
            .flat_map(|word| word.into_bits())
            .collect::<Vec<Boolean>>();
        pack_into_inputs(
            cs.namespace(|| "exposing result of sha256 processing"),
            output.as_slice(),
        )?;
        println!("synthesize call");

        Ok(())
    }
}

#[test]
fn end_to_end_sha256_test() {
    fn positive_test(preimage: Vec<u8>, expected_hash: Vec<u8>, use_circuit_prover: bool) {
        println!("positive test");
        assert!(test(preimage, expected_hash, use_circuit_prover));
        println!("#############");
    }

    fn negative_test(preimage: Vec<u8>, expected_hash: Vec<u8>, use_circuit_prover: bool) {
        println!("negative test");
        assert!(!test(preimage, expected_hash, use_circuit_prover));
        println!("#############");
    }

    fn test(preimage: Vec<u8>, expected_hash: Vec<u8>, use_circuit_prover: bool) -> bool {
        let circuit_verifier = Sha256Circuit::circuit_verifier();
        let circuit_prover = Sha256Circuit::circuit_prover(preimage);

        let public_inputs_prover = circuit_prover.compute_public_inputs(expected_hash.clone());
        let public_inputs_verifier = circuit_verifier.compute_public_inputs(expected_hash);
        assert_eq!(public_inputs_prover, public_inputs_verifier);

        let mut rng = OsRng;

        let parameters: Parameters<Bls12> = if use_circuit_prover {
            generate_random_parameters(circuit_prover.clone(), &mut rng)
                .expect("can't generate global parameters (circuit prover)")
        } else {
            generate_random_parameters(circuit_verifier.clone(), &mut rng)
                .expect("can't generate global parameters (circuit verifier)")
        };

        let verifier_key = prepare_verifying_key(&parameters.vk);

        let proof = create_random_proof(circuit_prover, &parameters, &mut rng)
            .expect("can't generate proof");

        verify_proof(&verifier_key, &proof, &public_inputs_prover).expect("can't verify proof")
    }

    let preimage: Vec<u8> = vec![0x00; 128];
    let mut expected_hash = sha256(preimage.clone());
    expected_hash.reverse();
    positive_test(preimage.clone(), expected_hash.clone(), false);
    positive_test(preimage, expected_hash, true);

    let preimage: Vec<u8> = vec![0xd8; 128];
    let mut expected_hash = sha256(preimage.clone());
    expected_hash.reverse();
    positive_test(preimage.clone(), expected_hash.clone(), false);
    positive_test(preimage, expected_hash, true);

    let preimage: Vec<u8> = vec![0xff; 128];
    let mut expected_hash = sha256(preimage.clone());
    expected_hash.reverse();
    expected_hash[0] = 0x00; // altering correct result
    negative_test(preimage.clone(), expected_hash.clone(), false);
    negative_test(preimage, expected_hash, true);
}

// just to check that self-made sha256 implementation works as expected
fn sha256(message: Vec<u8>) -> Vec<u8> {
    let message: [u8; PREIMAGE_LENGTH_BITS / 8] =
        message.try_into().unwrap_or_else(|v: Vec<u8>| {
            panic!(
                "Expected a Vec of length {} but it was {}",
                PREIMAGE_LENGTH_BITS / 8,
                v.len()
            )
        });

    let mut sha256 = Sha256::new();
    sha256.update(message.clone());

    let expected_hash: Vec<u8> = sha256.finalize().to_vec();
    let hash = sha256_hashing(message).to_vec();
    assert_eq!(expected_hash, hash);

    return hash;
}
