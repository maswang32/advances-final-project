import torch
from advances_project.training.trainer import Trainer

if __name__ == "__main__":
    save_dir = "../exp/test"
    eval_masks = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
        device="cuda",
    )
    trainer = Trainer(save_dir, eval_masks=eval_masks)
    trainer.train_loop()
