use std::io;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

use log::debug;
use pairing::MultiMillerLoop;
use supraseal_c2::SRS;

use crate::groth16::{ParameterSource, VerifyingKey};
use crate::SynthesisError;

// The parameters for Supraseal live on the C++ side. We take this just as a wrapper so that their
// are properly initialized.
pub struct SuprasealParameters<E> {
    ///// The parameter file we're reading from.
    //pub param_file_path: PathBuf,
    pub srs: SRS,
    _phantom: PhantomData<E>,
}

unsafe impl<E> Sync for SuprasealParameters<E> {}
unsafe impl<E> Send for SuprasealParameters<E> {}

impl<E> SuprasealParameters<E> {
    pub fn new(param_file_path: PathBuf) -> io::Result<Self> {
        //// Initialize the parameters on the C++ side.
        //supraseal_c2::read_srs(
        //    param_file_path
        //        .to_str()
        //        .expect("path must be valid UTF-8")
        //        .to_string(),
        //);
        debug!("Using parameter file at {:?}", param_file_path);
        let srs = SRS::try_new(param_file_path).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, String::from(err)))?;

        Ok(Self {
            //param_file_path,
            srs,
            _phantom: PhantomData::<E>,
        })
    }
}

impl<'a, E> ParameterSource<E> for &'a SuprasealParameters<E>
where
    E: MultiMillerLoop,
{
    type G1Builder = (Arc<Vec<E::G1Affine>>, usize);
    type G2Builder = (Arc<Vec<E::G2Affine>>, usize);

    fn get_vk(&self, _: usize) -> Result<&VerifyingKey<E>, SynthesisError> {
        unimplemented!()
    }

    fn get_h(&self, _num_h: usize) -> Result<Self::G1Builder, SynthesisError> {
        unimplemented!()
    }

    fn get_l(&self, _num_l: usize) -> Result<Self::G1Builder, SynthesisError> {
        unimplemented!()
    }

    fn get_a(
        &self,
        _num_inputs: usize,
        _num_a: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        unimplemented!()
    }

    fn get_b_g1(
        &self,
        _num_inputs: usize,
        _num_b_g1: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        unimplemented!()
    }

    fn get_b_g2(
        &self,
        _num_inputs: usize,
        _num_b_g2: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        unimplemented!()
    }
    fn get_supraseal_srs(&self) -> Option<supraseal_c2::SRS> {
        println!("vmx: bellperson: groth16: supraseal get srs was called");
        Some(self.srs.clone())
    }
}
