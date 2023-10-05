extern crate rust_cuda;

use rust_cuda::prelude::*;
use std::time::{Instant, Duration};
use std::io;

// Define a simple kernel that does some computation
#[global]
fn intensive_kernel(data: DeviceBox<[f32]>, n: usize) {
    let idx = blockIdx().x * blockDim().x + threadIdx().x;
    let mut f = idx as f32;
    for _ in 0..n {
        f = f.sin() * f.cos() * f.tan();  // Intensive math operations
        if idx < n {
            data[idx] += f;  // Use the allocated memory to prevent it from being optimized away
        }
    }
}

fn main() {
    let device_count = Device::count().unwrap();

    if device_count == 0 {
        println!("No GPUs found.");
        return;
    }

    for i in 0..device_count {
        let device = Device::get(i).unwrap();
        let prop = device.get_properties().unwrap();
        println!("Device {}: {}", i, prop.name);
        println!("  Compute capability: {}.{}", prop.major, prop.minor);
        println!("  Total global memory: {} bytes", prop.total_global_mem);
        println!("  Max threads per block: {}", prop.max_threads_per_block);
    }

    // Set the device to the first GPU
    Device::set_current(&Device::get(0).unwrap()).unwrap();

    // Ask the user for the amount of memory to allocate
    println!("Enter the amount of memory to allocate (in GBs): ");
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let memory_in_gb: f32 = input.trim().parse().unwrap();

    // Allocate the specified amount of GPU RAM
    let data_size = (memory_in_gb * 1024.0 * 1024.0 * 1024.0) as usize;  // Convert GB to bytes
    let d_data = DeviceBox::new(vec![0.0f32; data_size / 4]).unwrap();  // 4 bytes per float

    // Time the kernel execution
    let start = Instant::now();

    // Launch the kernel multiple times to ensure it runs for at least 5-10 minutes
    while start.elapsed() < Duration::from_secs(600) {  // 600 seconds = 10 minutes
        intensive_kernel<<<1_000_000, 256>>>(d_data, data_size / 4);  // Adjust the loop count inside the kernel if needed
        Device::current().unwrap().synchronize().unwrap();
    }

    // Kernel execution complete
    println!("Kernel execution complete.");
}
