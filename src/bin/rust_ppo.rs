use anyhow::Result;
use burn::backend::cuda::CudaDevice;
use burn::collective::{register, CollectiveConfig, PeerId};
use clap::Parser;

use rust_ppo::config::{Args, DistInfo};
use rust_ppo::models::InnerBackend;
use rust_ppo::ppo::train;

fn main() -> Result<()> {
    let args = Args::parse();
    let dist = DistInfo::from_env_or_args(&args)?;

    let device_index = if dist.world_size > 1 {
        dist.local_rank
    } else {
        args.cuda_device
    };
    let device = CudaDevice::new(device_index);

    if dist.world_size > 1 {
        let peer_id = PeerId::from(dist.rank);
        register::<InnerBackend>(
            peer_id,
            device.clone(),
            CollectiveConfig::default().with_num_devices(dist.world_size),
        )
        .map_err(|e| anyhow::anyhow!("failed to register burn collective: {e:?}"))?;
    }

    train::run(args, dist, device)
}
