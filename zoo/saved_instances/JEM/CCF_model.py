import torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
import zoo.saved_instances.JEM.wideresnet as wideresnet
import torch


class gradient_attack_wrapper(nn.Module):
  def __init__(self, model):
      super(gradient_attack_wrapper, self).__init__()
      self.model = model.eval()

  def forward(self, x):
      # mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1)).to('cuda')
      # sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1)).to('cuda')
      # print(x[0,:3,:3,0])
      # x = x * sigma
      # x = x + mean
      # print(x[0,:3,:3,0])
      # exit()
      x = x - 0.5
      x = x / 0.5
      x.requires_grad_()
      # out = self.model.refined_logits(x)
      out = self.model.logits(x)
      return out

  def eval(self):
    return self.model.eval()


class WrapperModel(nn.Module):
    def __init__(self, f, steps_to_refine):
        super(WrapperModel, self).__init__()
        self.f = f
        self.steps_to_refine = steps_to_refine

    def logits(self, x):
        return self.f.classify(x)

    def refined_logits(self, x):
        n_dup_chains = 5
        sigma = .03
        xs = x.size()
        dup_x = x.view(xs[0], 1, xs[1], xs[2], xs[3]).repeat(1, n_dup_chains, 1, 1, 1)
        dup_x = dup_x.view(xs[0] * n_dup_chains, xs[1], xs[2], xs[3])
        dup_x = dup_x + torch.randn_like(dup_x) * sigma
        refined = self.refine(dup_x, detach=False)
        # refined = dup_x
        logits = self.logits(refined)
        logits = logits.view(x.size(0), n_dup_chains, logits.size(1))
        logits = logits.mean(1)
        return logits

    def forward(self, x):
        # return self.refined_logits(x)
        return self.logits(x)

    def classify(self, x):
        logits = self.logits(x)
        pred = logits.max(1)[1]
        return pred

    def logpx_score(self, x):
        # unnormalized logprob, unconditional on class
        return self.f(x)

    def refine(self, x, detach=True):
        # runs a markov chain seeded at x, use n_steps=10
        n_steps = self.steps_to_refine
        x_k = torch.autograd.Variable(x, requires_grad=True) if detach else x
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(self.f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * torch.randn_like(x_k)
        final_samples = x_k.detach() if detach else x_k
        return final_samples

    def grad_norm(self, x):
        x_k = torch.autograd.Variable(x, requires_grad=True)
        f_prime = torch.autograd.grad(self.f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    def logpx_delta_score(self, x):
        n_steps = self.steps_to_refine
        # difference in logprobs from input x and samples from a markov chain seeded at x
        #
        init_scores = self.f(x)
        x_r = self.refine(x)
        final_scores = self.f(x_r)
        # for real data final_score is only slightly higher than init_score
        return init_scores - final_scores

    def logp_grad_score(self, x):
        return -self.grad_norm(x)




class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z)

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z)


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes)

    def forward(self, x, y=None):
        x = x - 0.5
        x = x / 0.5
        logits = self.classify(x)
        return logits
        # if y is None:
        #     return logits.logsumexp(1)
        # else:
        #     return torch.gather(logits, 1, y[:, None])
