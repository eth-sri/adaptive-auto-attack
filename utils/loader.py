import os
import torch
import torchvision
import torchvision.transforms as transforms
import os


IMAGENET_DIR = '/home/mislav/data/imagenet/'
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
ROOT_DIR = os.path.join(dname, "data")
# ROOT_DIR = '/media/chengyuan/wpc/meta-attacks/data/'


def get_mean_sigma(device, dataset):
    if dataset == 'cifar10':
        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))
        sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1))
    elif dataset == 'imagenet32' or dataset == 'imagenet64':
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
        sigma = torch.FloatTensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))
    else:
        mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
        sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
    return mean.to(device), sigma.to(device)


def get_mnist():
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root=ROOT_DIR, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root=ROOT_DIR, train=False, download=True, transform=transform_test)
    return train_set, test_set, 28, 1, 10


def get_fashion():
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(root=ROOT_DIR, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.FashionMNIST(root=ROOT_DIR, train=False, download=True, transform=transform_test)
    return train_set, test_set, 28, 1, 10


def get_svhn():
    train_set = torchvision.datasets.SVHN(root=ROOT_DIR, split='train', download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.SVHN(root=ROOT_DIR, split='test', download=True, transform=transforms.ToTensor())
    return train_set, test_set, 32, 3, 10


def get_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root=ROOT_DIR, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=ROOT_DIR, train=False, download=True, transform=transform_test)
    return train_set, test_set, 32, 3, 10


def get_imagenet(dataset):
    dim = 32 if dataset == 'imagenet32' else 64
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Resize(dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_valid = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(dim),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(IMAGENET_DIR, 'train')
    valid_dir = os.path.join(IMAGENET_DIR, 'val')

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    valid_set = torchvision.datasets.ImageFolder(valid_dir, transform=transform_valid)
    return train_set, valid_set, dim, 3, 1000


def get_loaders(dataset, train_batch=64, test_batch=64):
    if dataset == 'cifar10':
        train_set, test_set, input_size, input_channels, n_class = get_cifar10()
    elif dataset == 'mnist':
        train_set, test_set, input_size, input_channels, n_class = get_mnist()
    elif dataset == 'fashion':
        train_set, test_set, input_size, input_channels, n_class = get_fashion()
    elif dataset == 'svhn':
        train_set, test_set, input_size, input_channels, n_class = get_svhn()
    else:
        raise NotImplementedError('Unknown dataset')
    # return len(train_set), train_set, test_set, input_size, input_channels, n_class
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=8, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch, shuffle=False, num_workers=8, drop_last=False)
    return len(train_set), train_loader, test_loader, input_size, input_channels, n_class


def get_data_tensor(loader, num=10000, offset=0, device='cpu'):
    x = torch.tensor([], dtype=torch.float32)
    y = torch.tensor([], dtype=torch.long)
    ct = 0
    for i_batch, o_batch in loader:
        x = torch.cat([x, i_batch], dim=0)
        y = torch.cat([y, o_batch], dim=0)
        ct += int(i_batch.shape[0])
        if ct > num+offset:
            break
    return x[offset:num+offset].to(device), y[offset:num+offset].to(device)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensorX, tensorY):
        self.tensorX = tensorX
        self.tensorY = tensorY

    def __getitem__(self, index):
        return self.tensorX[index], self.tensorY[index]

    def __len__(self):
        return self.tensorX.shape[0]