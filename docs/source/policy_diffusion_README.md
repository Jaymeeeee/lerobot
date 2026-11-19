## Paper

https://diffusion-policy.cs.columbia.edu

## Denoising backbone

This local fork supports two action denoisers under the same `diffusion` policy type:

- `denoiser_type="unet"` (default): the existing conditional 1D U-Net. Configurations saved before this option was added continue to select U-Net.
- `denoiser_type="dit"`: a diffusion Transformer with one token per action timestep, temporal positional encoding, bidirectional self-attention, and MLP blocks. The diffusion timestep and encoded observations condition the blocks through AdaLN-Zero, following the [DiT architecture](https://github.com/facebookresearch/DiT).

Both backbones predict tensors with shape `(batch, horizon, action_dim)`. The selected `prediction_type` controls whether the target is noise (`epsilon`, the default) or clean actions (`sample`). DDPM/DDIM sampling, observation history, action queuing, normalization, and the loss retain their existing behavior.

The DiT option keeps the existing ResNet/SpatialSoftmax visual encoders and their RGB-D camera pairing. Visual features and robot state provide global conditioning; this implementation does not introduce cross-attention over image patches. Existing SpatialSoftmax visualizations still describe visual pooling, not the Transformer's temporal attention.

For a new training run, add these arguments to the usual dataset and training arguments:

```bash
--policy.type=diffusion \
--policy.denoiser_type=dit \
--policy.dit_hidden_dim=256 \
--policy.dit_num_layers=6 \
--policy.dit_num_heads=8 \
--policy.dit_mlp_ratio=4.0 \
--policy.dit_dropout=0.0
```

The values shown are the DiT defaults. `dit_hidden_dim` must be divisible by `dit_num_heads`. The shared `diffusion_step_embed_dim` controls the diffusion timestep embedding, and `gradient_checkpointing=true` enables checkpointing of the Transformer blocks during training. DiT accepts horizons that are not multiples of the U-Net downsampling factor. Keep `n_action_steps <= horizon - n_obs_steps + 1` when choosing the execution window.

Changing from U-Net to DiT requires training a new policy: the denoising weights have different structures. Start with a fresh output directory rather than changing an existing checkpoint's configuration. The saved configuration records the backbone, so normal `DiffusionPolicy.from_pretrained(...)` loading reconstructs it without an inference-time architecture override. In the MuJoCo project, the existing policy adapter and rollout commands load this saved configuration as well. Continue using the configured RGB-D `depth_camera_map` and metric depth inputs for RGB-D training and inference.

## Training

The reference implementation maintains an exponential moving average (EMA) of the policy weights during training and evaluates the EMA weights. To reproduce this behavior, enable the trainer's EMA:

```bash
lerobot-train \
  --policy.type=diffusion \
  --ema.type=warmup \
  ...
```

Checkpoints then contain a directly loadable copy of the EMA weights next to the live ones, e.g. for evaluation:

```bash
lerobot-eval --policy.path=outputs/train/.../checkpoints/last/pretrained_model_ema ...
```

The `warmup` type (`--ema.inv_gamma`, `--ema.power`, ...) uses the reference implementation's values. Other types are a constant decay (`--ema.type=constant --ema.decay=0.99`, as in openpi's pi0/pi05 training) and the EDM2 power function (`--ema.type=power --ema.sigma_rel=0.1`).

## Citation

```bibtex
@article{chi2024diffusionpolicy,
	author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
	title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	journal = {The International Journal of Robotics Research},
	year = {2024},
}
```
