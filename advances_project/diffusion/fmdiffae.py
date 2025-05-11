import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from advances_project.arc.encoder import Encoder
from advances_project.arc.fftmask import FFTMask


class FMDiffAE:
    def __init__(self, feature_map_channels=4, device="cuda"):
        self.feature_map_channels = feature_map_channels
        self.device = device

        self.encoder = Encoder(out_channels=feature_map_channels).to(self.device)
        self.fftmask = FFTMask().to(self.device)
        self.load_stable_diffusion()

        with torch.no_grad():
            # Store Embeddings for the model to use
            null_prompt_tokens = self.tokenizer(
                [""], padding="max_length", return_tensors="pt"
            ).to(device)

            self.null_prompt_embs = self.text_encoder(
                **null_prompt_tokens
            ).last_hidden_state.to(device)

    def load_stable_diffusion(
        self,
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        cache_dir="/data/scratch/ycda/cache/huggingface",
        first_layer_weights_path="/data/hai-res/ycda/advances_project/advances_project/diffusion/unet_convin_init_weights.pt",
    ):
        with torch.no_grad():
            self.vae = (
                AutoencoderKL.from_pretrained(
                    model_id, subfolder="vae", cache_dir=cache_dir
                )
                .to(self.device)
                .eval()
                .requires_grad_(False)
            )

            self.unet = UNet2DConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                cache_dir=cache_dir,
                in_channels=(4 + self.feature_map_channels),
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=False,
            )

            # Initialize the weights
            first_layer_weights = torch.load(first_layer_weights_path)
            self.unet.conv_in.weight.data[:, :4, :, :] = first_layer_weights
            self.unet.conv_in.weight.data[:, 4:, :, :] = (
                torch.randn_like(first_layer_weights) * 1e-2
            )
            self.unet = self.unet.to(self.device)

            self.scheduler = DDPMScheduler.from_pretrained(
                model_id, subfolder="scheduler", cache_dir=cache_dir
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14", cache_dir=cache_dir
            )
            self.text_encoder = (
                CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14", cache_dir=cache_dir
                )
                .to(self.device)
                .eval()
                .requires_grad_(False)
            )

    def forward(self, x):
        bs = x.shape[0]

        # Extract Latent, add noise
        with torch.no_grad():
            # Get Latent
            z = self.vae.encode(x).latent_dist.sample()
            z = z * self.vae.config.scaling_factor

            ts = torch.randint(
                0, self.scheduler.num_train_timesteps, (bs,), device=z.device
            )
            noise = torch.randn_like(z)
            z_noisy = self.scheduler.add_noise(z, noise, ts)

        # Expand Text Embeddings
        embs = self.null_prompt_embs.repeat(bs, 1, 1)

        # Get Feature Map, randomly bandpass
        feature_map = self.encoder(x)
        feature_map = self.fftmask(feature_map)

        # Concatenate Feature Map to noisy latent along channel dim
        net_in = torch.cat([z_noisy, feature_map], dim=1)

        (noise_pred,) = self.unet(
            net_in,
            ts,
            encoder_hidden_states=embs,
            return_dict=False,
        )
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def generate(
        self,
        inputs,
        masks,
        num_steps=50,
        cfg_scale=2,
    ):
        bs = inputs.shape[0]
        device = inputs.device

        # Expand Text Embeddings, 2x for CFG
        embs = self.null_prompt_embs.repeat(2 * bs, 1, 1)

        # Get Feature Map
        feature_map = self.encoder(inputs)
        feature_map = self.fftmask(feature_map, provided_group_masks=masks)
        feature_map = torch.cat([feature_map, torch.zeros_like(feature_map)], dim=0)

        latents = torch.randn((bs, 4, 64, 64), device=device, dtype=torch.float32)

        self.scheduler.set_timesteps(num_steps)

        for t in tqdm(self.scheduler.timesteps):
            latent_2x = torch.cat([latents, latents], dim=0)
            latent_2x = self.scheduler.scale_model_input(latent_2x, t)

            net_in = torch.cat([latent_2x, feature_map], dim=1)

            noise_pred = self.unet(
                net_in,
                t,
                encoder_hidden_states=embs,
            ).sample

            noise_cond, noise_uncond = noise_pred.chunk(2)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / self.vae.config.scaling_factor
        decoded = self.vae.decode(latents).sample
        return decoded
