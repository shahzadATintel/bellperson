//! Prover implementation implemented using SupraSeal (C++).

use std::{sync::Arc, time::Instant};

use ff::{Field, PrimeField};
use log::info;
use pairing::MultiMillerLoop;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use super::{ParameterSource, Proof, ProvingAssignment};
use crate::{
    gpu::GpuName, Circuit, ConstraintSystem, Index, SynthesisError, Variable, BELLMAN_VERSION,
};

#[allow(clippy::type_complexity)]
pub(super) fn create_proof_batch_priority_inner<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    randomization: Option<(Vec<E::Fr>, Vec<E::Fr>)>,
    _priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    info!(
        "Bellperson {} with SupraSeal is being used!",
        BELLMAN_VERSION
    );

    let (start, provers, input_assignments_no_repr, aux_assignments_no_repr) =
        synthesize_circuits_batch(circuits)?;

    let input_assignment_len = input_assignments_no_repr[0].len();
    let aux_assignment_len = aux_assignments_no_repr[0].len();
    let n = provers[0].a.len();
    let a_aux_density_total = provers[0].a_aux_density.get_total_density();
    let b_input_density_total = provers[0].b_input_density.get_total_density();
    let b_aux_density_total = provers[0].b_aux_density.get_total_density();
    let num_circuits = provers.len();

    let (r_s, s_s) = randomization.unwrap_or((
        vec![E::Fr::ZERO; num_circuits],
        vec![E::Fr::ZERO; num_circuits],
    ));

    // Make sure all circuits have the same input len.
    for prover in &provers {
        assert_eq!(
            prover.a.len(),
            n,
            "only equaly sized circuits are supported"
        );
        debug_assert_eq!(
            a_aux_density_total,
            prover.a_aux_density.get_total_density(),
            "only identical circuits are supported"
        );
        debug_assert_eq!(
            b_input_density_total,
            prover.b_input_density.get_total_density(),
            "only identical circuits are supported"
        );
        debug_assert_eq!(
            b_aux_density_total,
            prover.b_aux_density.get_total_density(),
            "only identical circuits are supported"
        );
    }

    let mut input_assignments_ref = Vec::with_capacity(num_circuits);
    let mut aux_assignments_ref = Vec::with_capacity(num_circuits);
    for i in 0..num_circuits {
        input_assignments_ref.push(input_assignments_no_repr[i].as_ptr());
        aux_assignments_ref.push(aux_assignments_no_repr[i].as_ptr());
    }

    let mut a_ref = Vec::with_capacity(num_circuits);
    let mut b_ref = Vec::with_capacity(num_circuits);
    let mut c_ref = Vec::with_capacity(num_circuits);

    for prover in &provers {
        a_ref.push(prover.a.as_ptr());
        b_ref.push(prover.b.as_ptr());
        c_ref.push(prover.c.as_ptr());
    }

    let mut proofs: Vec<Proof<E>> = Vec::with_capacity(num_circuits);
    // We call out to C++ code which is unsafe anyway, hence silence this warning.
    #[allow(clippy::uninit_vec)]
    unsafe {
        proofs.set_len(num_circuits);
    }

    let srs = params.get_supraseal_srs().ok_or_else(|| {
        log::error!("SupraSeal SRS wasn't allocated correctly");
        SynthesisError::MalformedSrs
    })?;
    supraseal_c2::generate_groth16_proof(
        a_ref.as_slice(),
        b_ref.as_slice(),
        c_ref.as_slice(),
        provers[0].a.len(),
        input_assignments_ref.as_mut_slice(),
        aux_assignments_ref.as_mut_slice(),
        input_assignment_len,
        aux_assignment_len,
        provers[0].a_aux_density.bv.as_raw_slice(),
        provers[0].b_aux_density.bv.as_raw_slice(),
        a_aux_density_total,
        b_input_density_total,
        b_aux_density_total,
        num_circuits,
        r_s.as_slice(),
        s_s.as_slice(),
        proofs.as_mut_slice(),
        srs,
    );

    let proof_time = start.elapsed();
    info!("prover time: {:?}", proof_time);

    Ok(proofs)
}

#[allow(clippy::type_complexity)]
fn synthesize_circuits_batch<Scalar, C>(
    circuits: Vec<C>,
) -> Result<
    (
        Instant,
        std::vec::Vec<ProvingAssignment<Scalar>>,
        std::vec::Vec<std::sync::Arc<std::vec::Vec<Scalar>>>,
        std::vec::Vec<std::sync::Arc<std::vec::Vec<Scalar>>>,
    ),
    SynthesisError,
>
where
    Scalar: PrimeField,
    C: Circuit<Scalar> + Send,
{
    let start = Instant::now();

    let mut provers = circuits
        .into_par_iter()
        .map(|circuit| -> Result<_, SynthesisError> {
            let mut prover = ProvingAssignment::new();

            prover.alloc_input(|| "", || Ok(Scalar::ONE))?;

            circuit.synthesize(&mut prover)?;

            for i in 0..prover.input_assignment.len() {
                prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
            }

            Ok(prover)
        })
        .collect::<Result<Vec<_>, _>>()?;

    info!("synthesis time: {:?}", start.elapsed());

    // Start fft/multiexp prover timer
    let start = Instant::now();
    info!("starting proof timer");

    let input_assignments_no_repr = provers
        .par_iter_mut()
        .map(|prover| {
            let input_assignment = std::mem::take(&mut prover.input_assignment);
            Arc::new(input_assignment)
        })
        .collect::<Vec<_>>();

    let aux_assignments_no_repr = provers
        .par_iter_mut()
        .map(|prover| {
            let aux_assignment = std::mem::take(&mut prover.aux_assignment);
            Arc::new(aux_assignment)
        })
        .collect::<Vec<_>>();

    Ok((
        start,
        provers,
        input_assignments_no_repr,
        aux_assignments_no_repr,
    ))
}
