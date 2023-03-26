import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

data_mean = (0.4914, 0.4822, 0.4465)
data_std = (0.2023, 0.1994, 0.2010)
batch_size = 800
kwargs = {
    'num_workers': 4,
    'pin_memory': True
}


def main():
    train_loader = DataLoader(
        dataset=CIFAR10('data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(
                                degrees=30, translate=(0.5, 0.5)),
                            transforms.ToTensor(),
                            transforms.Normalize(data_mean, data_std)
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    inputs_batch, labels_batch = next(iter(train_loader))
    grid = torchvision.utils.make_grid(inputs_batch, nrow=40, pad_value=1)
    torchvision.utils.save_image(grid, 'inputs_batch_preview.png')


if __name__ == '__main__':
    main()
