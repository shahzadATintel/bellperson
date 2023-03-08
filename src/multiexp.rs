use std::sync::Arc;

use ec_gpu_gen::multiexp_cpu::{multiexp_cpu, QueryDensity, SourceBuilder};
use ec_gpu_gen::threadpool::{Waiter, Worker};
use ec_gpu_gen::EcError;
use ff::PrimeField;
use group::prime::PrimeCurveAffine;

use crate::gpu;
pub use ec_gpu_gen::multiexp_cpu::DensityTracker;

/// Perform multi-exponentiation. The caller is responsible for ensuring the
/// query size is the same as the number of exponents.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub fn multiexp<'b, Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    kern: &mut gpu::LockedMultiexpKernel<G>,
) -> Waiter<Result<<G as PrimeCurveAffine>::Curve, EcError>>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine + gpu::GpuName,
    S: SourceBuilder<G>,
{
    // Try to run on the GPU.
    if let Ok(p) = kern.with(|k: &mut gpu::CpuGpuMultiexpKernel<G>| {
        let exps = density_map
            .as_ref()
            .generate_exps::<G::Scalar>(exponents.clone());
        let (bss, skip) = bases.clone().get();
        k.multiexp(pool, bss, exps, skip).map_err(Into::into)
    }) {
        return Waiter::done(Ok(p));
    }

    // Fallback to the CPU in case the GPU run failed.
    let result_cpu = multiexp_cpu(pool, bases, density_map, exponents);

    // Do not give the control back to the caller till the multiexp is done. Once done the GPU
    // might again be free, so we can run subsequent calls on the GPU instead of the CPU again.
    let result = result_cpu.wait();

    Waiter::done(result)
}

#[cfg(not(any(feature = "cuda", feature = "opencl", feature = "sppark")))]
pub fn multiexp<'b, Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    _kern: &mut gpu::LockedMultiexpKernel<G>,
) -> Waiter<Result<<G as PrimeCurveAffine>::Curve, EcError>>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine,
    S: SourceBuilder<G>,
{
    multiexp_cpu(pool, bases, density_map, exponents)
}

#[cfg(feature = "sppark")]
pub fn multiexp<'b, Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents_orig: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    _kern: &mut gpu::LockedMultiexpKernel<G>,
) -> Waiter<Result<<G as PrimeCurveAffine>::Curve, EcError>>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine,
    S: SourceBuilder<G>,
{
    let exponents = density_map
        .as_ref()
        .generate_exps::<G::Scalar>(exponents_orig.clone());
    let (bases_arc, skip) = bases.clone().get();
    // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
    // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
    let bases_slice = &bases_arc[skip..(skip + exponents.len())];
    let exponents_slice = &exponents[..];

    // `sppark` MSM doesn't work with <= 512 elements and also only on G1. For the rest, use the
    // CPU implementation.
    if exponents_slice.len() > 512
        && std::any::TypeId::of::<G>() == std::any::TypeId::of::<blstrs::G1Affine>()
    {
        log::debug!("vmx: running multiexp on sppark: {}", exponents_slice.len());
        let bases_concrete = unsafe {
            (&bases_slice as *const &[G])
                .cast::<&[blstrs::G1Affine]>()
                .as_ref()
                .unwrap()
        };
        let exponents_concrete = unsafe {
            (&exponents_slice as *const &[<<G as PrimeCurveAffine>::Scalar as PrimeField>::Repr])
                .cast::<&[[u8; 32]]>()
                .as_ref()
                .unwrap()
        };

        let point = fil_sppark::multi_scalar_multiplication(bases_concrete, exponents_concrete);
        let point_associate = unsafe {
            (&point as *const blstrs::G1Projective)
                .cast::<<G as PrimeCurveAffine>::Curve>()
                .as_ref()
                .unwrap()
        };
        Waiter::done(Ok(*point_associate))
    } else {
        log::debug!("Falling back to to CPU multiexp");
        multiexp_cpu(pool, bases, density_map, exponents_orig)
    }
}
