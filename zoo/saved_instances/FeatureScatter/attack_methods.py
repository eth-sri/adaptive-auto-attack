import numpy as np
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attack_FeaScatter(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_FeaScatter, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        print(config)

    def forward(self,
                inputs,
                targets=None,
                attack=False,
                targeted_label=-1,
                batch_idx=0):

        assert(not attack)
        outputs = self.basic_net(inputs)
        return outputs

