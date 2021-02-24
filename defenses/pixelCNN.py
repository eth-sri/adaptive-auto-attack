import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn
# from .defense_module import DefenseModule
import numpy as np


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
                log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))


def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
                log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda: one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def sample_from_discretized_mix_logistic_1d(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]  # [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # for mean, scale

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out


def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(F.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


''' utilities for shifting the image around, efficient alternative to masking convolutions '''


def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


def load_part_of_model(model, path):
    params = torch.load(path)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try:
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                print(e)
                pass
    print('added %s of params:' % (added / float(len(model.state_dict().keys()))))


class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0] * shp[1] * shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1),
                 shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = nn.ZeroPad2d((int((filter_size[1] - 1) / 2),  # pad left
                                 int((filter_size[1] - 1) / 2),  # pad right
                                 filter_size[0] - 1,  # pad top
                                 0))  # pad down

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
               int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1),
                 shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1),
                 shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                            stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x


'''
skip connection parameter : 0 = no skip connection 
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''


class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)  # cuz of concat elu

        if skip_connection != 0:
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)

    def forward(self, og_x, a=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None:
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * F.sigmoid(b)
        return og_x + c3


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=0) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []
        
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=2) 
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))
        
        return u, ul
         

class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, 
                    resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' : 
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else : 
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters, 
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters, 
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])
        
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3), 
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters, 
                                            filter_size=(1,3), shift_output_down=True), 
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters, 
                                            filter_size=(2,1), shift_output_right=True)])
    
        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None


    def forward(self, x, sample=False):
        # similar as done in the tf repo :  
        if self.init_padding is None and not sample: 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding
        
        if sample : 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2: 
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()
        
        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out


def train():
    import time
    import os
    import argparse
    import torch
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torchvision import datasets, transforms, utils

    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='mnist', help='Can be either cifar|mnist')
    parser.add_argument('-p', '--print_every', type=int, default=50,
                        help='how many iterations between print statements')
    parser.add_argument('-t', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.99995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='Batch size during training per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=50, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)
    assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)

    obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
    input_channels = obs[0]
    rescaling = lambda x: (x - .5) * 2.
    rescaling_inv = lambda x: .5 * x + .5
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    if 'mnist' in args.dataset:
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True,
                                                                  train=True, transform=ds_transforms),
                                                   batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False,
                                                                 transform=ds_transforms),
                                                  batch_size=args.batch_size,
                                                  shuffle=True, **kwargs)

        loss_op = lambda real, fake: discretized_mix_logistic_loss_1d(real, fake)

    elif 'cifar' in args.dataset:
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True,
                                                                    download=True, transform=ds_transforms),
                                                   batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False,
                                                                   transform=ds_transforms),
                                                  batch_size=args.batch_size,
                                                  shuffle=True, **kwargs)

        loss_op = lambda real, fake: discretized_mix_logistic_loss(real, fake)
    else:
        raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                     input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    model = model.cuda()

    if args.load_params:
        load_part_of_model(model, args.load_params)
        # model.load_state_dict(torch.load(args.load_params))
        print('model parameters loaded')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    print('starting training')
    writes = 0
    for epoch in range(args.max_epochs):
        model.train(True)
        torch.cuda.synchronize()
        train_loss = 0.
        time_ = time.time()
        model.train()
        for batch_idx, (input, _) in enumerate(train_loader):
            input = input.cuda(async=True)
            input = Variable(input)
            output = model(input)
            loss = loss_op(input, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data
            if (batch_idx + 1) % args.print_every == 0:
                deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
                print('{:d} loss : {:.4f}, time : {:.4f}'.format(
                    epoch,
                    (train_loss / deno),
                    (time.time() - time_)))
                train_loss = 0.
                writes += 1
                time_ = time.time()

        # decrease learning rate
        scheduler.step()
        torch.cuda.synchronize()
        model.eval()
        test_loss = 0.
        for batch_idx, (input, _) in enumerate(test_loader):
            input = input.cuda(async=True)
            input_var = Variable(input)
            output = model(input_var)
            loss = loss_op(input_var, output)
            test_loss += loss.data
            del loss, output

        deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
        print('test loss : %s' % (test_loss / deno))

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
        

if __name__ == '__main__':
    # ''' testing loss with tf version '''
    # np.random.seed(1)
    # xx_t = (np.random.rand(15, 32, 32, 100) * 3).astype('float32')
    # yy_t  = np.random.uniform(-1, 1, size=(15, 32, 32, 3)).astype('float32')
    # x_t = Variable(torch.from_numpy(xx_t)).cuda()
    # y_t = Variable(torch.from_numpy(yy_t)).cuda()
    # loss = discretized_mix_logistic_loss(y_t, x_t)
    #
    # ''' testing model and deconv dimensions '''
    # x = torch.cuda.FloatTensor(32, 3, 32, 32).uniform_(-1., 1.)
    # xv = Variable(x).cpu()
    # ds = down_shifted_deconv2d(3, 40, stride=(2,2))
    # x_v = Variable(x)
    #
    # ''' testing loss compatibility '''
    # model = PixelCNN(nr_resnet=3, nr_filters=100, input_channels=x.size(1))
    # model = model.cuda()
    # out = model(x_v)
    # loss = discretized_mix_logistic_loss(x_v, out)
    # print('loss : %s' % loss.data[0])
    train()



