use std::env;

use log::info;
#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;
use rust_gpu_tools::{Device, Framework, GPUError, Program};

#[cfg(not(all(feature = "cuda", feature = "opencl")))]
use crate::error::EcError;
use crate::error::EcResult;

/// Selects a CUDA or OpenCL on the `EC_GPU_FRAMEWORK` environment variable and the
/// compile-time features.
///
/// You cannot select CUDA if the library was compiled without support for it.
#[allow(clippy::unnecessary_wraps)] // No error can be returned if `cuda` and `opencl `are enabled.
fn select_framework(default_framework: Framework) -> EcResult<Framework> {
    match env::var("EC_GPU_FRAMEWORK") {
        Ok(env) => match env.as_ref() {
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Framework::Cuda)
                }

                #[cfg(not(feature = "cuda"))]
                Err(EcError::Simple("CUDA framework is not supported, please compile with the `cuda` feature enabled."))
            }
            "opencl" => {
                #[cfg(feature = "opencl")]
                {
                    Ok(Framework::Opencl)
                }

                #[cfg(not(feature = "opencl"))]
                Err(EcError::Simple("OpenCL framework is not supported, please compile with the `opencl` feature enabled."))
            }
            _ => Ok(default_framework),
        },
        Err(_) => Ok(default_framework),
    }
}

/// Returns the program for the preferred [`rust_gpu_tools::Framework`].
///
/// If the device supports CUDA, then CUDA is used, else OpenCL. You can force a selection with
/// the environment variable `EC_GPU_FRAMEWORK`, which can be set either to `cuda` or `opencl`.
pub fn program(device: &Device) -> EcResult<Program> {
    let framework = select_framework(device.framework())?;
    program_use_framework(device, &framework)
}

/// Returns the program for the specified [`rust_gpu_tools::Framework`].
pub fn program_use_framework(device: &Device, framework: &Framework) -> EcResult<Program> {
    match framework {
        #[cfg(feature = "cuda")]
        Framework::Cuda => {
            info!("Using kernel on CUDA.");
            let kernel = include_bytes!(env!("CUDA_KERNEL_FATBIN"));
            let cuda_device = device.cuda_device().ok_or(GPUError::DeviceNotFound)?;
            let program = cuda::Program::from_bytes(cuda_device, kernel)?;
            Ok(Program::Cuda(program))
        }
        #[cfg(feature = "opencl")]
        Framework::Opencl => {
            info!("Using kernel on CUDA.");
            let kernel = include_bytes!(env!("CUDA_KERNEL_FATBIN"));
            let cuda_device = device.cuda_device().ok_or(GPUError::DeviceNotFound)?;
            let program = cuda::Program::from_bytes(cuda_device, kernel)?;
            Ok(Program::Cuda(program))
        }
    }
}
