import os
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from advances_project.diffusion.fmdiffae import FMDiffAE
from advances_project.data.artbench import get_loader


def log_mem(tag=""):
    mb = torch.cuda.memory_reserved() / 2**20
    print(f"{tag:<12} reserved = {mb:,.1f} MB")


class Trainer:
    def __init__(
        self, save_dir, device="cuda", eval_masks=[], num_epochs=1000, load_path=None
    ):
        self.save_dir = save_dir
        self.device = device
        self.eval_masks = eval_masks
        self.num_epochs = num_epochs

        self.fmdiffae = FMDiffAE(device=device)
        self.init_dataloaders()
        self.init_optimizer()
        self.init_logging()

        if load_path is not None:
            self.load_checkpoint(load_path)

    def init_dataloaders(self):
        self.train_dataloader = get_loader(split="train")
        self.valid_dataloader = get_loader(split="valid")

    def init_optimizer(self):
        self.optimizer = Adam(
            [
                {"params": self.fmdiffae.unet.parameters(), "lr": 1e-5},
                {"params": self.fmdiffae.encoder.parameters(), "lr": 1e-4},
            ]
        )

    def init_logging(self):
        self.step = 0
        self.train_losses = []
        self.valid_losses = []
        self.recon_losses = []
        self.ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        self.output_dir = os.path.join(self.save_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    @torch.no_grad()
    def valid_loss(self, num_iters=16):
        tmp_valid_losses = []

        for i, batch in tqdm(
            enumerate(self.valid_dataloader), total=num_iters, desc="Validating"
        ):
            if i >= num_iters:
                break

            tmp_valid_losses.append(
                self.fmdiffae.forward(batch[0].to(self.device)).item()
            )

        self.valid_losses.append([self.step, np.mean(tmp_valid_losses)])

    @torch.no_grad()
    def generate_examples(self, batch_size=8):
        inputs = next(iter(self.valid_dataloader))[0][0].to(self.device)
        print("Input shape", inputs.shape)

        generated = []
        for batch_eval_masks in self.eval_masks.split(batch_size):
            generated.append(
                self.fmdiffae.generate(
                    inputs.expand(batch_eval_masks.shape[0], -1, -1, -1),
                    masks=batch_eval_masks,
                )
            )
        generated = torch.cat(generated, dim=0)
        print("Generated shape", generated.shape)

        self.recon_losses.append(
            [
                (
                    ((generated - inputs[None, ...]) ** 2)
                    .mean(dim=(1, 2, 3))
                    .cpu()
                    .numpy()
                    .tolist()
                ),
            ]
        )

        # Plotting
        gen_vis = (generated * 0.5 + 0.5).clamp(0, 1)
        fig, axs = plt.subplots(4, 4, figsize=(16, 16))
        for ax, img_tensor, mask in zip(axs.flatten(), gen_vis, self.eval_masks):
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            bits = mask.cpu().numpy().astype(int).tolist()
            mstr = "".join(str(b) for b in bits)
            ax.set_title(mstr, fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{self.step:06d}.png"), dpi=300)
        plt.close(fig)

        # Cleaning Up
        del generated, inputs
        gc.collect()
        torch.cuda.empty_cache()

    def eval_loop(self):
        self.valid_loss()
        log_mem("Before Example Generation")
        self.generate_examples()
        log_mem("After Example Generation")
        self.save_checkpoint()

    def train_step(self, batch):
        loss = self.fmdiffae.forward(batch[0].to(self.device))
        self.train_losses.append([self.step, loss.item()])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_loop(self):
        self.fmdiffae.unet.train()

        for epoch in range(self.num_epochs):
            print(f"\nEpoch: {epoch}")
            for batch in tqdm(self.train_dataloader):
                self.train_step(batch)

                if self.step % 1000 == 0:
                    self.fmdiffae.unet.eval()
                    self.eval_loop()
                    self.fmdiffae.unet.train()

                self.step = self.step + 1

    def save_checkpoint(self):
        ckpt = {
            "step": self.step,
            "unet_state": self.fmdiffae.unet.state_dict(),
            "encoder_state": self.fmdiffae.encoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(self.ckpt_dir, f"{self.step}.pt"))

        np.save(os.path.join(self.save_dir, "train_losses.npy"), self.train_losses)
        np.save(os.path.join(self.save_dir, "valid_losses.npy"), self.valid_losses)
        np.save(os.path.join(self.save_dir, "recon_losses.npy"), self.recon_losses)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)

        self.fmdiffae.unet.load_state_dict(ckpt["unet_state"])
        self.fmdiffae.encoder.load_state_dict(ckpt["encoder_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step = ckpt["step"]

        train_loss_path = os.path.join(self.save_dir, "train_losses.npy")
        valid_loss_path = os.path.join(self.save_dir, "valid_losses.npy")
        recon_loss_path = os.path.join(self.save_dir, "recon_losses.npy")

        if os.path.exists(train_loss_path):
            self.train_losses = np.load(train_loss_path, allow_pickle=True).tolist()
        if os.path.exists(valid_loss_path):
            self.valid_losses = np.load(valid_loss_path, allow_pickle=True).tolist()
        if os.path.exists(recon_loss_path):
            self.recon_losses = np.load(recon_loss_path, allow_pickle=True).tolist()

        print(f"[Checkpoint] Loaded checkpoint from {path}")
