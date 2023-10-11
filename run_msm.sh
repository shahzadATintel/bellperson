source ./env_setup.sh
#RUSTFLAGS="-C target-cpu=native" cargo test --release --all --features cuda,opencl
#RUST_LOG=info cargo test --features cuda,opencl -- --exact groth16::multiscalar::tests::test_multiscalar_par --nocapture
RUST_LOG=info cargo test --features cuda,opencl -- --exact test_parallel_prover --nocapture
