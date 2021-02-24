import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.classifier import Classifier
from utils.dag_module import DAGModule
from utils.loader import get_mean_sigma
from .layers import Conv2d, Normalization, ReLU, Flatten, Linear, Sequential


class SeqNet(nn.Module):

    def __init__(self):
        super(SeqNet, self).__init__()
        self.is_double = False
        self.skip_norm = False

    def forward(self, x, init_lambda=False):
        if isinstance(x, torch.Tensor) and self.is_double:
            x = x.to(dtype=torch.float64)
        x = self.blocks(x, init_lambda, skip_norm=self.skip_norm)
        return x

    def reset_bounds(self):
        for block in self.blocks:
            block.bounds = None

    def to_double(self):
        self.is_double = True
        for param_name, param_value in self.named_parameters():
            param_value.data = param_value.data.to(dtype=torch.float64)

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.blocks.forward_until(i, x)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.blocks.forward_from(i, x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = False


class FFNN(SeqNet):

    def __init__(self, device, dataset, sizes, n_class=10, input_size=32, input_channel=3):
        super(FFNN, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [Flatten(), Linear(input_size * input_size * input_channel, sizes[0]), ReLU(sizes[0])]
        for i in range(1, len(sizes)):
            layers += [
                Linear(sizes[i - 1], sizes[i]),
                ReLU(sizes[i]),
            ]
        layers += [Linear(sizes[-1], n_class)]
        self.blocks = Sequential(*layers)


class ConvMed(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1,
                 linear_size=100):
        super(ConvMed, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16 * width1, 5, stride=2, padding=2, dim=input_size),
            ReLU((16 * width1, input_size // 2, input_size // 2)),
            Conv2d(16 * width1, 32 * width2, 4, stride=2, padding=1, dim=input_size // 2),
            ReLU((32 * width2, input_size // 4, input_size // 4)),
            Flatten(),
            Linear(32 * width2 * (input_size // 4) * (input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)


class ConvMedBatchNorm(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1,
                 linear_size=100):
        super(ConvMedBatchNorm, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16 * width1, 5, stride=2, padding=2, dim=input_size),
            ReLU((16 * width1, input_size // 2, input_size // 2)),
            nn.BatchNorm2d(16 * width1),
            Conv2d(16 * width1, 32 * width2, 4, stride=2, padding=1, dim=input_size // 2),
            ReLU((32 * width2, input_size // 4, input_size // 4)),
            nn.BatchNorm2d(32 * width2),
            Flatten(),
            Linear(32 * width2 * (input_size // 4) * (input_size // 4), linear_size),
            ReLU(linear_size),
            nn.BatchNorm1d(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)


class ConvMedBig(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=4, width2=4, width3=2,
                 linear_size=200, with_normalization=True):
        super(ConvMedBig, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        if with_normalization:
            layers = [
                Normalization(mean, sigma),
                Conv2d(input_channel, 16 * width1, 3, stride=1, padding=1, dim=input_size),
                ReLU((16 * width1, input_size, input_size)),
                Conv2d(16 * width1, 16 * width2, 4, stride=2, padding=1, dim=input_size // 2),
                ReLU((16 * width2, input_size // 2, input_size // 2)),
                Conv2d(16 * width2, 32 * width3, 4, stride=2, padding=1, dim=input_size // 2),
                ReLU((32 * width3, input_size // 4, input_size // 4)),
                Flatten(),
                Linear(32 * width3 * (input_size // 4) * (input_size // 4), linear_size),
                ReLU(linear_size),
                Linear(linear_size, n_class),
            ]
        else:
            layers = [
                Conv2d(input_channel, 16 * width1, 3, stride=1, padding=1, dim=input_size),
                ReLU((16 * width1, input_size, input_size)),
                Conv2d(16 * width1, 16 * width2, 4, stride=2, padding=1, dim=input_size // 2),
                ReLU((16 * width2, input_size // 2, input_size // 2)),
                Conv2d(16 * width2, 32 * width3, 4, stride=2, padding=1, dim=input_size // 2),
                ReLU((32 * width3, input_size // 4, input_size // 4)),
                Flatten(),
                Linear(32 * width3 * (input_size // 4) * (input_size // 4), linear_size),
                ReLU(linear_size),
                Linear(linear_size, n_class),
            ]
        self.blocks = Sequential(*layers)
        self.layers = layers


class ConvMedBig1(SeqNet):
    def __init__(self, convmedbig):
        super(ConvMedBig1, self).__init__()
        assert(isinstance(convmedbig, ConvMedBig)), "This wrapper takes convmedbig model only"
        self.blocks = Sequential(*convmedbig.layers[:-2])


class ConvMedBig2(SeqNet):
    def __init__(self, convmedbig):
        super(ConvMedBig2, self).__init__()
        assert(isinstance(convmedbig, ConvMedBig)), "This wrapper takes convmedbig model only"
        self.blocks = Sequential(*convmedbig.layers[-2:])


class SimpleNet(Classifier):
    def __init__(self, in_ch, out_ch):
        super(SimpleNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_ch, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=out_ch)
        self.nclasses = out_ch

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

    def forwardToDetect(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        return x


class SimpleNet1(Classifier):
    def __init__(self, in_ch, out_ch):
        super(SimpleNet1, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_ch, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        # self.fc_2 = nn.Linear(in_features=100, out_features=out_ch)
        # self.nclasses = out_ch

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        return x


class SimpleNet2(Classifier):
    def __init__(self, in_ch, out_ch):
        super(SimpleNet2, self).__init__()
        self.fc_2 = nn.Linear(in_features=100, out_features=out_ch)
        self.nclasses = out_ch

    def forward(self, x):
        x = self.fc_2(x)
        return x


class SimpleDetector(Classifier):
    def __init__(self, dmodel, in_features=100):
        super(SimpleDetector, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=2)
        self.param = [self.fc1.weight, self.fc1.bias]

    def forward(self, x):
        x = self.fc1(x)
        return x


class SimpleEnsemble(Classifier):
    def __init__(self, in_ch, out_ch, N):
        super(SimpleEnsemble, self).__init__()
        self.N = N
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modelList = []
        for i in range(N):
            self.modelList.append(Classifier(SimpleNet(in_ch, out_ch)))

    def forward(self, x):
        pred = torch.zeros(x.shape[0], self.out_ch)
        for model in self.modelList:
            pred += model(x)
        return pred



