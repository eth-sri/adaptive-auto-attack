from time import time
import torch
import torch.nn as nn
from defenses import *
from utils.dag_module import DAGModule, Submodule
from utils.loader import get_data_tensor
import numpy as np
from art.utils import is_probability
from tqdm import trange
from torch.nn.modules.utils import _pair

from copy import deepcopy
from attack.attack_pool import LinfAPGDAttack, L2APGDAttack
from attack.attack_util import explode
from utils.eval import get_result_from_loader
import eagerpy as ep

device = 'cuda'
NUM_NETWORK_TEST_SAMPLES = 2000


def check_softmax(model, images):
    modelout = model(images).detach().cpu().numpy()
    is_softmax = True
    for i in range(modelout.shape[0]):
        is_softmax = is_probability(modelout[i, :]) & is_softmax
    return is_softmax


def custom_bpda_substitute(model, test_loader):
    bpdawrapperlist = []

    def thermometer_forwardsub(x):
        num_space = 10
        b, ch, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        device = x.device
        newx = torch.zeros([b, ch * num_space, h, w]).to(device)
        for j in range(ch):
            for i in range(num_space):
                thres = i / num_space
                newx[:, i+j*num_space:i+j*num_space+1, :] = torch.where(x[:, j:j+1, :, :] - thres > 0, x[:, j:j+1, :, :] - thres, torch.tensor(0.0).to(device))
        return newx

    for module in model.eval_model.moduleList:
        if isinstance(module, JpegCompression):
            bpdawrapperlist.append(BPDAWrapper(module, forwardsub=lambda x: x))
        elif isinstance(module, ThermometerEncoding):
            bpdawrapperlist.append(BPDAWrapper(module, forwardsub=thermometer_forwardsub))
        else:
            bpdawrapperlist.append(module)
    model.bpda_model = Submodule(bpdawrapperlist, model.dependency, model.output, model.test_fit)


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=True):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class SimpleApproxNetwork(nn.Module):
    def __init__(self, in_features, out_features, shape):
        super(SimpleApproxNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv_1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1)
        self.require_training = True

    def forward(self, x):
        return self.conv_1(x)


class LocallyApproxNetwork(nn.Module):
    def __init__(self, in_features, out_features, shape):
        super(LocallyApproxNetwork, self).__init__()
        self.conv = LocallyConnected2d(in_channels=in_features, out_channels=out_features,
                                       output_size=[shape[2], shape[3]], kernel_size=1, stride=1, bias=True)
        # self.conv_1 = nn.Conv2d(in_channels=in_features, out_channels=nhidden, kernel_size=3, stride=1, padding=1)
        # self.conv_2 = nn.Conv2d(in_channels=nhidden, out_channels=out_features, kernel_size=3, stride=1, padding=1)
        self.require_training = True

    def forward(self, x):
        return self.conv(x)


class ComplexApproxNetwork(nn.Module):
    def __init__(self, in_features, out_features, shape):
        super(ComplexApproxNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv_1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)
        self.require_training = True

    def forward(self, x):
        return self.conv_2(self.conv_1(x).relu())


class LargeApproxNetwork(nn.Module):
    def __init__(self, in_features, out_features, shape):
        super(LargeApproxNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv_1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)
        self.require_training = True

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x).relu()).relu())


class IdentityNetwork(nn.Module):
    def __init__(self, in_features, out_features, shape):
        super(IdentityNetwork, self).__init__()
        ratio = out_features/in_features
        assert(ratio == int(ratio)), "Identity BPDA requires out-feature to be integer multiple of in-feature"
        self.ratio = int(ratio)
        self.require_training = False

    def forward(self, x):
        # assume there are four dimensions
        return x.repeat(1, self.ratio, 1, 1)


class timer:
    def __init__(self, msg):
        self.msg = msg
        self.start = time()

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg, "{:4f} s".format(time()-self.start))


def bpda_training(x, xprox, network=SimpleApproxNetwork, ch_idx=1):
    """
    x is the data input with shape: batch, input channel, height, width.
    xprox is the output with shape: batch, output_channel, height, width
    """
    if(len(x.shape) > 2):
        assert(x.shape[2] == xprox.shape[2]), "bpda_approximation: the spatial dimension has to be same"
    if(len(x.shape) > 3):
        assert(x.shape[3] == xprox.shape[3]), "bpda_approximation: the spatial dimension has to be same"
    with timer("Time used for bpda"):
        import random
        size = x.shape[0]
        in_features = x.shape[ch_idx]
        out_features = xprox.shape[ch_idx]
        if isinstance(network, list):
            net = network.pop(0)
        else:
            net = network
        model = net(in_features, out_features, x.shape).to(x.device)
        if model.require_training:
            # initialization
            # for p in model.parameters():
            #     p.data.fill_(0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            loss_fcn = nn.MSELoss()
            batch_size = 64
            nb_epochs = 5
            num_batch = int(np.ceil(size / float(batch_size)))
            ind = np.arange(size)
            # Start training
            for _ in trange(nb_epochs, desc="Training bpda: "):
                random.shuffle(ind)
                for m in range(num_batch):
                    i_batch = x[ind[m * batch_size: (m + 1) * batch_size]]
                    o_batch = xprox[ind[m * batch_size: (m + 1) * batch_size]]
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Perform prediction
                    model_outputs = model(i_batch)
                    # Form the loss function
                    loss = loss_fcn(model_outputs, o_batch)
                    # Actual training
                    loss.backward()
                    optimizer.step()
    return model


def create_module_differentiable(module, test_loader, network=SimpleApproxNetwork):
    device = 'cuda'
    testx, testy = get_data_tensor(test_loader, num=100000, device=device)
    xprox = module(testx).to(device)
    return bpda_training(testx, xprox, network)


def module_property_test(testx, module):
    testx.requires_grad_(True)
    testx.retain_grad()
    testy = module(testx)
    testp = torch.sum(testy)
    is_non_diff = 0
    is_random = False
    is_removable = False

    # check if two forward pass values are close
    if not torch.isclose(module(testx), module(testx)).all():
        is_random = True
    try:
        testp.backward()
        assert(testx.grad is not None)
    except RuntimeError as e:
        is_non_diff = 1

    if testx.shape == testy.shape:
        is_removable = True
    return [is_non_diff, is_random, is_removable]


def create_bpda_module(testx, module, test_loader, network=SimpleApproxNetwork):
    is_non_diff, is_random, is_removable = module_property_test(testx, module)
    if is_non_diff:
        print("Meet non-differentiable layer: ", type(module))
        forwardsub = create_module_differentiable(module, test_loader, network)
        return BPDAWrapper(module, forwardsub=forwardsub)
    return module


def do_layer_removal(testx, module, model, idx, strategy):
    is_non_diff, is_random, is_removable = module_property_test(testx, module)
    if is_removable:
        print("meet removable layer", type(module))
        if isinstance(strategy, list):
            to_remove = strategy.pop(0)
        else:
            to_remove = strategy
        if to_remove:
            model.vertex_removal(idx)


def auto_bpda_substitute(model, test_loader, network=SimpleApproxNetwork):
    bpdawrapperlist = []
    # testx = torch.FloatTensor(1, 1, 28, 28).uniform_(0, 1).to(device)
    testx, testy = get_data_tensor(test_loader, num=1, device='cuda')
    for i, module in enumerate(model.eval_model.moduleList):
        testx_f = model.forward_until(testx, model.dependency[i]).detach()  # forward until the dependency of the module
        if isinstance(module, DAGModule):
            auto_bpda_substitute(module, test_loader, network)
        bpdawrapperlist.append(create_bpda_module(testx_f, module, test_loader, network))
    model.bpda_model = Submodule(bpdawrapperlist, model.dependency, model.output, model.test_fit)


def vertex_removal(model, test_loader, strategy=False):
    testx, testy = get_data_tensor(test_loader, num=1, device='cuda')
    for i, module in enumerate(model.eval_model.moduleList):
        testx_f = model.forward_until(testx, model.dependency[i]).detach()
        if isinstance(module, DAGModule):
            vertex_removal(module, test_loader, strategy=strategy)
        do_layer_removal(testx_f, module, model, i, strategy)


class network_property:
    def __init__(self, result):
        self.num_non_diff = result[0]
        self.is_random = result[1]
        self.is_softmax = result[2]
        self.num_removable = result[3]
        self.detector = None
        self.gradient_mask = None

    def print(self):
        print("Number of non differentiable vertices: ", self.num_non_diff)
        print("Network randomized: ", self.is_random)
        print("Network output softmax: ", self.is_softmax)
        print("Number of removable layers: ", self.num_removable)


def defense_property_test(model, test_loader, verbose=False):
    testx, testy = get_data_tensor(test_loader, num=5, device='cuda')
    num_non_diff = 0
    num_removable = 0
    is_random_model = False
    if not isinstance(model, DAGModule):
        print("The model is not DAGModule, cannot test some defense_properties")
        module_num_non_diff, is_random, is_removable = module_property_test(testx, model)
        num_non_diff += module_num_non_diff
        num_removable += is_removable
        is_random_model = is_random | is_random_model
        return network_property([num_non_diff, is_random_model, 0, num_removable])
    for i, module in enumerate(model.eval_model.moduleList):
        testx_f = model.forward_until(testx, model.dependency[i]).detach()  # forward until the dependency of the module
        if isinstance(module, DAGModule):
            network_pro = defense_property_test(module, test_loader)
            module_num_non_diff = network_pro.num_non_diff
            is_random = network_pro.is_random
            is_removable = network_pro.num_removable
        else:
            module_num_non_diff, is_random, is_removable = module_property_test(testx_f, module)
        num_non_diff += module_num_non_diff
        num_removable += is_removable
        is_random_model = is_random | is_random_model
    is_softmax = check_softmax(model, testx)
    result = network_property([num_non_diff, is_random_model, is_softmax, num_removable])
    if verbose:
        result.print()
    return result


def bpda_strategy(norm, epsilon, f_net, test_loader, defense_property):
    """
    BPDA substitution main
    """
    num_non_diff = defense_property.num_non_diff

    if num_non_diff == 0:
        return f_net

    g_net = deepcopy(f_net)

    test_param = {"m_iter": 50, "seed": 10}

    if norm == ep.inf:
        test_attack = LinfAPGDAttack(test_param)
    elif norm == 2:
        test_attack = L2APGDAttack(test_param)
    else:
        assert False, "Not supported norm for bpda test attack"
    # strategies = [ComplexApproxNetwork]
    # strategies = [SimpleApproxNetwork, ComplexApproxNetwork]
    strategies = [IdentityNetwork, SimpleApproxNetwork, ComplexApproxNetwork]
    strategy_list = explode(strategies, num_non_diff)
    acc = []
    for strategy in strategy_list:
        print("BPDA Strategy is: ", strategy)
        auto_bpda_substitute(g_net.instance.model, test_loader, strategy)
        result = get_result_from_loader([test_attack], [g_net], test_loader, epsilon, NUM_NETWORK_TEST_SAMPLES, norm,
                                        to_print=False, eval_net=f_net)
        print("The score is: ", result[0][0][1])
        acc.append(result[0][0][1])  # get the attack success rate
    idx = np.argmax(acc)
    strategy_list = explode(strategies, num_non_diff)
    best_strategy = strategy_list[idx]
    print(best_strategy)
    auto_bpda_substitute(g_net.instance.model, test_loader, best_strategy)
    return g_net


def removal_strategy(norm, epsilon, in_net, test_loader, defense_property, eval_net=None):
    """
    Layer removal main
    """
    num_removable = defense_property.num_removable
    if num_removable == 0:
        return in_net

    g_net = deepcopy(in_net)

    test_param = {"m_iter": 50, "seed": 10}

    if norm == ep.inf:
        test_attack = LinfAPGDAttack(test_param)
    elif norm == 2:
        test_attack = L2APGDAttack(test_param)
    else:
        assert False, "Not supported norm for bpda test attack"
    strategies = [1, 0]
    strategy_list = explode(strategies, num_removable)
    acc = []
    for strategy in strategy_list:
        print("Removal strategy is: ", strategy)
        vertex_removal(g_net.instance.model, test_loader, strategy)
        if eval_net is None:
            eval_net = in_net
        result = get_result_from_loader([test_attack], [g_net], test_loader, epsilon, NUM_NETWORK_TEST_SAMPLES, norm,
                                        to_print=False, eval_net=eval_net)
        acc.append(result[0][0][1])  # get the attack success rate
        print("The ASR is: ", result[0][0][1])
        g_net.instance.model.dep_copy_apply()
    idx = np.argmax(acc)
    strategy_list = explode(strategies, num_removable)
    best_strategy = strategy_list[idx]
    print(best_strategy)
    vertex_removal(g_net.instance.model, test_loader, best_strategy)
    return g_net


def search_g_net(norm, epsilon, f_net, test_loader, defense_property, use_vertex_removal):
    """
    Network transformation main
    """
    g_net = bpda_strategy(norm, epsilon, f_net, test_loader, defense_property)
    if use_vertex_removal:
        g_net = removal_strategy(norm, epsilon, g_net, test_loader, defense_property, eval_net=f_net)
    return g_net
