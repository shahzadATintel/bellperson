export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export BELLMAN_CPU_UTILIZATION=0
export BELLMAN_VERIFIER=gpu
export RUST_GPU_TOOLS_CUSTOM_GPU="Tesla T4:2560"
export BELLMAN_GPU_FRAMEWORK=cuda
export BELLMAN_NO_GPU=0
rustup override set 1.66.0
