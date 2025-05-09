import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from advances_project.data.artbench import get_loader
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import Adam


def load_models(
    model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    device="cuda",
    cache_dir="/data/scratch/ycda/advances_project/huggingface",
):
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", cache_dir=cache_dir
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", cache_dir=cache_dir
    ).to(device)

    scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir=cache_dir
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=cache_dir
    )

    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=cache_dir
    ).to(device)

    return vae, unet, scheduler, tokenizer, text_encoder


@torch.no_grad()
def generate(
    vae,
    unet,
    scheduler,
    prompt_embeddings,
    tokenizer,
    text_encoder,
    batch_size=16,
    num_steps=50,
    cfg_scale=7,
    save_path="output.png",
    device="cuda",
):
    uncond_tokens = tokenizer(
        [""] * batch_size, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)

    uncond_embs = text_encoder(**uncond_tokens).last_hidden_state
    embeddings = torch.cat([uncond_embs, prompt_embeddings[:batch_size]])

    latents = torch.randn(
        (batch_size, unet.in_channels, 64, 64), device=device, dtype=torch.float32
    )

    scheduler.set_timesteps(num_steps)
    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=embeddings,
        ).sample

        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    decoded = vae.decode(latents).sample
    decoded = (decoded * 0.5 + 0.5).clamp(0, 1)

    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    for ax, img_tensor in zip(axs.flatten(), decoded):
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved image grid to {save_path}")


if __name__ == "__main__":
    device = "cuda"
    batch_size = 16
    lr = 1e-5
    num_epochs = 1
    prompt = "activated pencil dog"
    resume_ckpt = None

    # Load Models
    vae, unet, scheduler, tokenizer, text_encoder = load_models(device=device)
    vae.eval()
    text_encoder.eval()
    unet.train()
    num_steps = scheduler.num_train_timesteps

    # Freeze VAE and text_encoder
    for p in vae.parameters():
        p.requires_grad = False

    for p in text_encoder.parameters():
        p.requires_grad = False

    optimizer = Adam(unet.parameters(), lr=lr)

    dataloader = get_loader(split="train", batch_size=batch_size)

    # Get embeddings
    tokens = tokenizer(
        [prompt] * batch_size,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    embs = text_encoder(**tokens).last_hidden_state.to(device)

    if resume_ckpt and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        unet.load_state_dict(ckpt["unet_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Resumed from {resume_ckpt}")

    step = 0
    losses = []
    for epoch in range(num_epochs):
        for img, _ in tqdm(dataloader):
            with torch.no_grad():
                # Get Latent
                z = vae.encode(img.to(device)).latent_dist.sample()
                z = z * vae.config.scaling_factor

                current_bs = z.shape[0]

                ts = torch.randint(0, num_steps, (current_bs,), device=device)
                noise = torch.randn_like(z)
                noisy_latents = scheduler.add_noise(z, noise, ts)

            (noise_pred,) = unet(
                noisy_latents,
                ts,
                encoder_hidden_states=embs[:current_bs],
                return_dict=False,
            )
            loss = F.mse_loss(noise_pred, noise)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 100 == 0:
                ckpt = {
                    "epoch": epoch,
                    "step": step,
                    "unet_state": unet.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                path = os.path.join("exp/latest.pt")
                torch.save(ckpt, path)
                print(f"[Checkpoint] Saved {path}")

                np.save("exp/losses.npy", losses)

                generate(
                    vae=vae,
                    unet=unet,
                    scheduler=scheduler,
                    prompt_embeddings=embs,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    batch_size=16,
                    save_path=f"exp/{step:06d}.png",
                    device=device,
                )

            step += 1
