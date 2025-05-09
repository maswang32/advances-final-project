from torchvision import transforms
from torch.utils.data import Dataset


class NormalizedDataset(Dataset):
    def __init__(self, dataset, res=512):
        self.dataset = dataset

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    res, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(res),
                transforms.ToTensor(),  # [0,1]
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __getitem__(self, index):
        item, target = self.dataset[index]
        item = self.transform(item)
        return item, target

    def __len__(self):
        return len(self.dataset)
