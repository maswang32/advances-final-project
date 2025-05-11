import torch
from torchvision.datasets import LSUN
from torch.utils.data import DataLoader, random_split
from advances_project.data.data_utils import NormalizedDataset


def get_loader(
    root="/data/hai-res/shared/temp/datasets/art",
    batch_size=32,
    split="train",
    num_workers=4,
    res=512,
    classes=[
        "art_nouveau",
        "baroque",
        "expressionism",
        "impressionism",
        "post_impressionism",
        "realism",
        "renaissance",
        "romanticism",
        "surrealism",
        "ukiyo_e",
    ],
):
    LSUN._verify_classes = lambda self, classes: list(classes)
    dataset = NormalizedDataset(LSUN(classes=classes, root=root), res=res)

    split_indices = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }

    generator = torch.Generator().manual_seed(42)
    dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)[
        split_indices[split]
    ]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=True,
    )
