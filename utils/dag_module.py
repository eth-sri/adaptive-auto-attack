""" This module implements DAGModule which defines the DAG graph of the input network

DAGModule instantiates nn.Module. It is used to define the custom DAG graph for the network. Usually it is used if
there are some input processing stage, post processing stage or some ensemble components. Currently the functionality
is still limited.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict

from defenses import *
from copy import deepcopy
from math import ceil


def get_module_name(module):
    return str(type(module)).split("'")[1].split(".")[-1]


class Submodule(nn.Module):
    def __init__(self, moduleList, dependency, output, fitlist):
        super(Submodule, self).__init__()
        self.moduleList = nn.ModuleList(moduleList)
        self.dependency = dependency
        # self.output = output
        self.fitlist = fitlist
        self.size = len(moduleList)
        # for idx, module in enumerate(moduleList):
        #     if fitlist[idx]:
        #         self.add_module(str(idx), module)

    def moduleForward(self, idx, cache):
        # input layer is -1 and it is the base case
        if idx == -1:
            return cache[-1]
        modulein = []
        for i in self.dependency[idx]:
            if not self.fitlist[i] and i != -1:
                ct = i
                while not self.fitlist[ct]:
                    assert(len(self.dependency[ct]) == 1), "skip fit should depend on only one input"
                    ct = self.dependency[ct][0]
                    if ct == -1:
                        break
                cidx = ct
            else:
                cidx = i
            if cache[cidx] is not None:
                modulein.append(cache[cidx])
            else:
                modulein.append(self.moduleForward(cidx, cache))
        if len(modulein) == 1:
            modulein = modulein[0]
        else:
            modulein = torch.stack(modulein, dim=1)
        result = self.moduleList[idx](modulein)
        cache[idx] = result
        return result

    def forward(self, x, output):
        outlist = []
        outidxlist = []
        cache = [None] * (len(self.moduleList) + 1)
        cache[-1] = x
        for i in output:
            ct = i
            processed = False
            if ct == -1:
                outidxlist.append(ct)
            else:
                while not self.fitlist[ct]:
                    if (len(self.dependency[ct]) == 1):
                        ct = self.dependency[ct][0]
                    else:
                        for k in self.dependency[ct]:
                            assert(self.fitlist[k]), "does not support multiple dependency with fit module to be False"
                            outidxlist.append(k)
                            processed = True
                        break
                if not processed:
                    outidxlist.append(ct)
        for ct in outidxlist:
            outlist.append(self.moduleForward(ct, cache))
        if len(outlist) == 1:
            return outlist[0]
        return outlist


class DAGModule(nn.Module):
    def __init__(self, moduleList, dependency=None, output=None, device='cuda'):
        super(DAGModule, self).__init__()
        self.defense_count = 0
        self.device = device
        self.moduleList = moduleList
        module_dict = OrderedDict()
        self.dependency = []
        self.train_fit = []
        self.test_fit = []
        self.output = []
        self.training = True
        if dependency is None:
            ct = -1
            for _ in moduleList:
                self.dependency.append([ct])
                ct += 1
        else:
            self.dependency = dependency

        if output is None:
            self.output = [len(self.moduleList)-1]
        else:
            self.output = output

        self.depend_copy = deepcopy(self.dependency)
        self.output_copy = deepcopy(self.output)
        # dependency list check
        assert(len(self.moduleList) == len(self.dependency))
        assert(len(self.output))

        for idx, module in enumerate(moduleList):
            if isinstance(module, DefenseModule):
                pre_defense_wrap = BPDAWrapper(module)
                # name = get_module_name(module)
                name = str(idx)
                assert (name not in module_dict)
                module_dict[name] = pre_defense_wrap
                if module.apply_fit:
                    self.train_fit.append(True)
                else:
                    self.train_fit.append(False)
                if module.apply_predict:
                    self.test_fit.append(True)
                else:
                    self.test_fit.append(False)
                self.defense_count += 1
            else:
                assert(isinstance(module, nn.Module)), "The module has to be torch.nn.module"
                self.train_fit.append(True)
                self.test_fit.append(True)
        self.train_model = Submodule(self.moduleList, self.dependency, self.output, self.train_fit).to(device)
        self.eval_model = Submodule(self.moduleList, self.dependency, self.output, self.test_fit).to(device)
        self.bpda_model = None
        # self.bpda_substitute()

    def dep_copy_apply(self):
        self.dependency = deepcopy(self.depend_copy)
        self.train_model.dependency = self.dependency
        self.eval_model.dependency = self.dependency
        if self.bpda_model is not None:
            self.bpda_model.dependency = self.dependency
        self.set_output(deepcopy(self.output_copy))
        for module in self.moduleList:
            if isinstance(module, DAGModule):
                module.dep_copy_apply()

    def set_output(self, value):
        self.output = value
        self.train_model.output = value
        self.eval_model.output = value
        if self.bpda_model is not None:
            self.bpda_model.output = value

    def vertex_removal(self, idx):
        assert(len(self.dependency[idx]) == 1)
        prev_idx = self.dependency[idx][0]
        self.dependency[idx] = []

        # dependency handover
        for i in range(len(self.dependency)):
            for j in range(len(self.dependency[i])):
                if self.dependency[i][j] == idx:
                    self.dependency[i][j] = prev_idx
        # output reassign
        for i in range(len(self.output)):
            if self.output[i] == idx:
                self.output[i] = prev_idx

    def train(self, mode=True):
        for module in self.moduleList:
            if isinstance(module, DAGModule):
                module.training = mode
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        if self.training:
            return self.train_model(x, self.output)
        else:
            if self.bpda_model is not None:
                return self.bpda_model(x, self.output)
            else:
                return self.eval_model(x, self.output)

    def forward_until(self, x, output):
        cp = self.output
        self.set_output(output)

        result = self.forward(x)

        self.set_output(cp)
        # HACK for handling multiple input items
        if isinstance(result, list) and len(result) > 1:
            result = torch.stack(result, dim=1)
        return result

    def fit(self, loader, loss_fcn, optimizer, nb_epochs=10, **kwargs):
        normal_loss = False
        if isinstance(loss_fcn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss)):
            normal_loss = True
        # Start training
        optimizer.zero_grad()
        for i in range(nb_epochs):
            pbar = tqdm(loader)
            for i_batch, o_batch in pbar:
                i_batch, o_batch = i_batch.to('cuda'), o_batch.to('cuda')
                # Perform prediction
                model_outputs = self.forward(i_batch)
                # HACK
                if (type(model_outputs) == type([]) and normal_loss):
                    model_outputs = model_outputs[0]
                # Form the loss function
                loss = loss_fcn(model_outputs, o_batch)
                loss.backward()
                optimizer.step()
                pbar.set_description("epoch {:d}: loss {:.3f}".format(i+1, loss))
                optimizer.zero_grad()

    def advfit(self, loader, loss_fcn, optimizer, epsilon, attack=None, nb_epochs=10, ratio=0.5):
        if attack is None:
            return self.fit(loader, loss_fcn, optimizer, nb_epochs=nb_epochs)
        assert (0 <= ratio <= 1), "ratio must be between 0 and 1"

        warmstart = nb_epochs//5 # default warm start to normal train the network
        self.fit(loader, loss_fcn, optimizer, nb_epochs=warmstart)
        normal_loss = False
        if isinstance(loss_fcn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss)):
            normal_loss = True

        # Start training
        for i in range(warmstart, nb_epochs):
            pbar = tqdm(loader)
            # Shuffle the examples
            optimizer.zero_grad()
            for i_batch, o_batch in pbar:
                i_batch, o_batch = i_batch.to('cuda'), o_batch.to('cuda')

                self.eval()
                # fmodel = fb.PyTorchModel(self, bounds=(0, 1))
                adv_batch, _ = attack(self, i_batch, o_batch, epsilon)
                self.train()
                optimizer.zero_grad()
                # Perform prediction, if using torch loss and if there are multiple items in the list, take the first one
                # HACK
                model_outputs = self.forward(i_batch)
                if (type(model_outputs) == type([]) and normal_loss):
                    model_outputs = model_outputs[0]
                adv_outputs = self.forward(adv_batch)
                if (type(adv_outputs) == type([]) and normal_loss):
                    adv_outputs = adv_outputs[0]
                loss = (1 - ratio) * loss_fcn(model_outputs, o_batch) + ratio * loss_fcn(adv_outputs, o_batch)
                # Actual training
                loss.backward()
                optimizer.step()
                pbar.set_description("epoch {:d}: loss {:.3f}".format(i+1, loss))
                optimizer.zero_grad()

    def save(self, filename, path):
        import os
        assert (path is not None)

        full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.state_dict(), full_path + ".model")

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            param.detach()

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = False

    # def bpda_substitute(self):
    #     bpdawrapperlist = []
    #
    #     def thermometer_forwardsub(x):
    #         num_space = 10
    #         b, h, w = x.shape[0], x.shape[2], x.shape[3]
    #         device = x.device
    #         newx = torch.zeros([b, num_space, h, w]).to(device)
    #         for i in range(num_space):
    #             thres = i / num_space
    #             newx[:, i:i + 1, :] = torch.where(x - thres > 0, x - thres, torch.tensor(0.0).to(device))
    #         return newx
    #
    #     for module in self.eval_model.moduleList:
    #         if isinstance(module, JpegCompression):
    #             bpdawrapperlist.append(BPDAWrapper(module, forwardsub=lambda x: x))
    #         elif isinstance(module, ThermometerEncoding):
    #             bpdawrapperlist.append(BPDAWrapper(module, forwardsub=thermometer_forwardsub))
    #         else:
    #             bpdawrapperlist.append(module)
    #     self.bpda_model = Submodule(bpdawrapperlist, self.dependency, self.output, self.test_fit)


class DistillationWrapper(DAGModule):
    def __init__(self, model, temperature):
        super(DistillationWrapper, self).__init__(moduleList=[model])
        self.temperature = temperature

    def forward(self, x):
        return super(DistillationWrapper, self).forward(x) / self.temperature