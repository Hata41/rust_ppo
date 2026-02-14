use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;

#[derive(Module, Clone, Debug)]
pub struct Mlp<Bk: burn::tensor::backend::Backend> {
    l1: Linear<Bk>,
    l2: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Mlp<Bk> {
    pub fn new(in_dim: usize, hidden: usize, device: &Bk::Device) -> Self {
        let l1 = LinearConfig::new(in_dim, hidden).init(device);
        let l2 = LinearConfig::new(hidden, hidden).init(device);
        Self { l1, l2 }
    }

    pub fn forward(&self, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let x = burn::tensor::activation::relu(self.l1.forward(x));
        burn::tensor::activation::relu(self.l2.forward(x))
    }
}

#[derive(Module, Clone, Debug)]
pub struct Actor<Bk: burn::tensor::backend::Backend> {
    torso: Mlp<Bk>,
    head: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Actor<Bk> {
    pub fn new(obs_dim: usize, hidden: usize, action_dim: usize, device: &Bk::Device) -> Self {
        let torso = Mlp::new(obs_dim, hidden, device);
        let head = LinearConfig::new(hidden, action_dim).init(device);
        Self { torso, head }
    }

    pub fn forward(&self, obs: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let h = self.torso.forward(obs);
        self.head.forward(h)
    }
}

#[derive(Module, Clone, Debug)]
pub struct Critic<Bk: burn::tensor::backend::Backend> {
    torso: Mlp<Bk>,
    head: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Critic<Bk> {
    pub fn new(obs_dim: usize, hidden: usize, device: &Bk::Device) -> Self {
        let torso = Mlp::new(obs_dim, hidden, device);
        let head = LinearConfig::new(hidden, 1).init(device);
        Self { torso, head }
    }

    pub fn forward(&self, obs: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let h = self.torso.forward(obs);
        self.head.forward(h)
    }
}

#[derive(Module, Clone, Debug)]
pub struct Agent<Bk: burn::tensor::backend::Backend> {
    pub actor: Actor<Bk>,
    pub critic: Critic<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Agent<Bk> {
    pub fn new(obs_dim: usize, hidden: usize, action_dim: usize, device: &Bk::Device) -> Self {
        Self {
            actor: Actor::new(obs_dim, hidden, action_dim, device),
            critic: Critic::new(obs_dim, hidden, device),
        }
    }
}