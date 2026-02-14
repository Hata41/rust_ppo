use anyhow::{bail, Result};
use burn::backend::autodiff::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn::collective::{register, CollectiveConfig, PeerId};
use burn_ndarray::{NdArray, NdArrayDevice};
use clap::Parser;

use rust_ppo::config::{Args, DeviceType, DistInfo};
use rust_ppo::ppo::train;

fn main() -> Result<()> {
    let args = Args::parse();
    let dist = DistInfo::from_env_or_args(&args)?;

    match args.device_type {
        DeviceType::Cpu => {
            if dist.world_size > 1 {
                bail!(
                    "--device-type cpu does not support distributed training (WORLD_SIZE must be 1)"
                );
            }
            train::run::<Autodiff<NdArray<f32>>>(args, dist, NdArrayDevice::Cpu)
        }
        DeviceType::Cuda => {
            let device_index = if dist.world_size > 1 {
                dist.local_rank
            } else {
                args.cuda_device
            };
            let device = CudaDevice::new(device_index);

            if dist.world_size > 1 {
                let peer_id = PeerId::from(dist.rank);
                register::<Cuda<f32, i32>>(
                    peer_id,
                    device.clone(),
                    CollectiveConfig::default().with_num_devices(dist.world_size),
                )
                .map_err(|e| anyhow::anyhow!("failed to register burn collective: {e:?}"))?;
            }

            train::run::<Autodiff<Cuda<f32, i32>>>(args, dist, device)
        }
    }
}
