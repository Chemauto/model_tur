import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_REGISTRY = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


def get_dataloaders(cfg):
    name = cfg["dataset"]
    spec = cfg["datasets"][name]
    c = spec["channels"]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * c, [0.5] * c),  # 归一化到 [-1, 1], 与 DDPM 约定一致
    ])
    cls = _REGISTRY[name]
    train_ds = cls("./data", train=True, download=True, transform=tf)
    test_ds = cls("./data", train=False, download=True, transform=tf)
    kwargs = dict(num_workers=cfg["train"]["num_workers"], pin_memory=torch.cuda.is_available())
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          drop_last=True, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], **kwargs)
    return train_dl, test_dl
