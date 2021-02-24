import torch
import torch.nn as nn
from utils.classifier import Classifier

# class PLogDet(Function):
#   @staticmethod
#   def forward(ctx, x):
#     l = torch.cholesky(x)
#     ctx.save_for_backward(l)
#     return 2 * l.diagonal(dim1=-2, dim2=-1).log().sum(-1)
#
#   @staticmethod
#   def backward(ctx, g):
#     l, = ctx.saved_tensors
#     n = l.shape[-1]
#     # use cholesky_inverse once pytorch/pytorch/issues/7500 is solved
#     return g * torch.cholesky_solve(torch.eye(n, out=l.new(n, n)), l)

# plogdet = PLogDet.apply
log_offset = 1e-20
det_offset = 1e-6


def Entropy(pred):
    #input shape is batch_size X num_class
    # return tf.reduce_sum(-tf.multiply(input, tf.log(input + log_offset)), axis=-1)
    return torch.sum(-pred * (torch.log(pred + log_offset)), dim=-1)


def log_det(y_true, y_pred, num_model):
    num_classes = 10
    bool_R_y_true = torch.ne(torch.ones(y_true.shape) - y_true, 0)  # batch_size X (num_class X num_models), 2-D
    mask_non_y_pred = y_pred[bool_R_y_true]  # batch_size X (num_class-1) X num_models, 1-D
    mask_non_y_pred = torch.reshape(mask_non_y_pred, [-1, num_model, num_classes - 1])  # batch_size X num_model X (num_class-1), 3-D
    mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, dim=2, keepdim=True)  # batch_size X num_model X (num_class-1), 3-D
    matrix = torch.bmm(mask_non_y_pred, torch.transpose(mask_non_y_pred, 1, 2))  # batch_size X num_model X num_model, 3-D
    all_log_det = torch.log(torch.linalg.det(matrix + det_offset * torch.unsqueeze(torch.eye(num_model), 0)))  # batch_size X 1, 1-D

    # bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, zero) # batch_size X (num_class X num_models), 2-D
    # mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
    # mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
    # mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
    # matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
    # all_log_det = tf.linalg.logdet(matrix+det_offset*tf.expand_dims(tf.eye(num_model),0)) # batch_size X 1, 1-D
    return all_log_det


class DiversityEnsembleLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=1):
        super(DiversityEnsembleLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta


    def forward(self, inputs, target):
        lossfcn = nn.CrossEntropyLoss()
        predfcn = nn.Softmax(dim=1)
        loss = None
        predList = []
        # model loss
        if type(inputs) != type([]):
            inputs = [inputs]
        for unit in inputs:
            pred = predfcn(unit)  # class probability
            predList.append(pred)
            if loss is None:
                loss = lossfcn(unit, target)
            else:
                loss += lossfcn(unit, target)
        # entropy metric
        for pred in predList:
            loss += self.alpha * torch.mean(Entropy(pred))
        return loss


# class DiversityEnsemble(Classifier):
#     def __init__(self, modelList, alpha=0, beta=0):
#         super(DiversityEnsemble, self).__init__()
#         self.modelList = nn.ModuleList(modelList)
#         self.nmodel = len(modelList)
#         self.alpha = alpha
#         self.beta = beta
#         self.nclasses = modelList[0].nclasses
#
#     def forward(self, x):
#         pred = torch.zeros(x.shape[0], self.nclasses).to(x.device)
#         for model in self.modelList:
#             pred += model.forward(x)
#         return pred
#
#     def ensembleLoss(self, x, label):
#         lossfcn = nn.NLLLoss()
#         loss = 0
#         predList = []
#         # model loss
#         for model in self.modelList:
#             pred = nn.LogSoftmax(model(x)) # class probability
#             predList.append(pred)
#             loss += lossfcn(pred, label)
#         # entropy metric
#         for pred in predList:
#             loss += torch.mean(Entropy(pred))
#         # det metric
#         # for pred in predList:
#         #     loss += torch.mean(log_det(label, pred, self.nmodel))
#
#     def fit(self, x, y, loss_fcn, optimizer, batch_size=128, nb_epochs=10, **kwargs):
#         return super(DiversityEnsemble, self).fit(x, y, loss_fcn, optimizer, batch_size=128, nb_epochs=10, **kwargs)

