import torch
import copy

from tqdm import tqdm
from advances_project.data.artbench import get_loader
from advances_project.training.trainer import Trainer

if __name__ == "__main__":
    # Load Trainer
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
            [0, 0, 1, 1, 1, 1],
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
    trainer = Trainer(
        save_dir,
        eval_masks=eval_masks,
        load_path="/data/hai-res/ycda/advances_project/exp/test/checkpoints/67000.pt",
    )

    # Get just the encoder
    net = copy.deepcopy(trainer.fmdiffae.encoder).cuda()
    del trainer
    torch.cuda.empty_cache()

    # Get Data
    valid_loader = get_loader(split="valid", drop_last=False)

    # Compute and save encodings
    with torch.no_grad():
        valid_encodings = []
        valid_labels = []
        for batch in tqdm(valid_loader):
            valid_encodings.append(net(batch[0].cuda()).cpu())
            valid_labels.append(batch[1])

    valid_encodings = torch.cat(valid_encodings, dim=0)
    valid_labels = torch.cat(valid_labels, dim=0)

    print(valid_encodings.shape)
    print(valid_labels.shape)

    torch.save(valid_encodings, "valid_encodings.pt")
    torch.save(valid_labels, "valid_labels.pt")
