""" This file implements the different loss functions. New loss functions can be added here

Loss function have signature: f, X, Y -> float,
where f is the model, X are the clean images, and Y are the labels.
Implementation is not as clean as the formulation, but it gives efficiency

get_loss_fn: the main method, returns a callable loss function.
The return callable takes in X as input. f, Y are inputs to the function.

The current implementation does not check if the model already returns probability,
and assumes the models output logits
"""

import eagerpy as ep
import numpy as np
from typing import Callable
from foolbox_custom.models.base import Model


def get_loss_fn(model: Model, labels: ep.Tensor, type: str, targeted: bool, modifier: dict
        ) -> Callable[[ep.Tensor, ep.Tensor], ep.Tensor]:
    """
    The main function to get the loss function
    model: model f
    labels: Y
    type: type of the loss (shown in loss_dict)
    targeted: the number of targeted is implemented in attack_base
    modifier: details in get_modified_loss

    The signature of the returned callable can be changed by flags in modifiers (details in get_modified_loss)
    """
    loss_dict = {
        "ce": get_ce_loss_fn,
        "l1": get_l1_loss_fn,
        "hinge": get_hinge_loss_fn,
        "dlr": get_dlr_loss_fn,
        "logit": get_logit_loss_fn,
    }
    if type in loss_dict:
        loss_fn = loss_dict[type](model, labels, targeted, modifier)
    else:
        assert False, "Unrecognized loss function"
    return loss_fn


def get_modified_loss(model, inputs, labels, untargeted_fn, targeted_fn, targeted, modifier):
    """
    Return the loss function based on the modifiers.
    There are five modifiers:
    1. softmax: flag to control if attack on raw outputs / the one with softmax
    2. loss_diff: flag to controls the targeted loss to be subtracted by the untargeted loss
    *3. indiv: modify the return to return individual losses instead of the sum of losses
    *4. logits: return the prediction (affected by labels)
    *5. labels: takes in the array of labels and return the logits of the labels and the logits of the top class (for SQR)
    Here 3, 4, 5 does not change the functionality, and they are used to change the return for algorithm implementation
    """
    logits = model(inputs)
    logits, restore_type = ep.astensor_(logits)

    outputs = logits
    if 'softmax' in modifier:
        if modifier['softmax']:
            outputs = logits.softmax()

    if targeted:
        if 'loss_diff' in modifier and modifier['loss_diff']:
            ind_sorted = outputs.argsort(axis=1)
            ind = (ind_sorted[:, -1])
            losses = targeted_fn(outputs, labels) + untargeted_fn(outputs, ind)
        else:
            losses = targeted_fn(outputs, labels)
        loss = losses.sum()
    else:
        losses = untargeted_fn(outputs, labels)
        loss = losses.sum()

    result = [restore_type(loss)]
    if 'indiv' in modifier:
        result.append(restore_type(losses))
    if 'logits' in modifier:
        result.append(restore_type(outputs))
    if 'labels' in modifier:
        curr_idx = modifier['labels']
        u = np.arange(labels.shape[0])
        y_corr = logits[u, curr_idx]
        logits.raw[u, curr_idx] = -float('inf')
        y_others = logits.max(axis=-1)
        result.append([restore_type(y_corr), restore_type(y_others)])
    if len(result) == 1:
        result = result[0]
    return result


def get_ce_loss_fn(model: Model, labels: ep.Tensor, targeted: bool, modifier: dict,
                   ) -> Callable[[ep.Tensor, ep.Tensor], ep.Tensor]:

    def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
        modifier['softmax'] = True   # overwrites softmax parameter
        indices = np.arange(labels.shape[0])

        targeted_fn = lambda z, labels: z[indices, labels].log()
        untargeted_fn = lambda z, labels: -z[indices, labels].log()

        return get_modified_loss(model, inputs, labels, untargeted_fn, targeted_fn, targeted, modifier)

    return loss_fn


def get_l1_loss_fn(model: Model, labels: ep.Tensor, targeted: bool, modifier: dict,
                   ) -> Callable[[ep.Tensor, ep.Tensor], ep.Tensor]:

    def loss_fn(inputs: ep.Tensor) -> ep.Tensor:

        indices = np.arange(labels.shape[0])

        targeted_fn = lambda z, labels: z[indices, labels]
        untargeted_fn = lambda z, labels: -z[indices, labels]

        return get_modified_loss(model, inputs, labels, untargeted_fn, targeted_fn, targeted, modifier)

    return loss_fn


def get_hinge_loss_fn(model: Model, labels: ep.Tensor, targeted: bool, modifier: dict,
                   ) -> Callable[[ep.Tensor, ep.Tensor], ep.Tensor]:

    def loss_fn(inputs: ep.Tensor) -> ep.Tensor:

        indices = np.arange(labels.shape[0])

        def targeted_fn(z, labels):
            z_sorted = z.sort(axis=1)
            ind_sorted = z.argsort(axis=1)
            ind = (ind_sorted[:, -1] == labels).float32()
            return z[indices, labels] - z_sorted[:, -2] * ind - z_sorted[:, -1] * (1. - ind)


        def untargeted_fn(z, labels):
            z_sorted = z.sort(axis=1)
            ind_sorted = z.argsort(axis=1)
            ind = (ind_sorted[:, -1] == labels).float32()
            return z_sorted[:, -2] * ind + z_sorted[:, -1] * (1. - ind) - z[indices, labels]

        return get_modified_loss(model, inputs, labels, untargeted_fn, targeted_fn, targeted, modifier)

    return loss_fn


def get_dlr_loss_fn(model: Model, labels: ep.Tensor, targeted: bool, modifier: dict,
                   ) -> Callable[[ep.Tensor, ep.Tensor], ep.Tensor]:

    def loss_fn(inputs: ep.Tensor) -> ep.Tensor:

        indices = np.arange(labels.shape[0])

        def targeted_fn(z, labels):
            z_sorted = z.sort(axis=1)
            ind_sorted = z.argsort(axis=1)
            ind = (ind_sorted[:, -1] == labels).float32()
            return -(z_sorted[:, -2] * ind + z_sorted[:, -1] * (1. - ind) - z[indices, labels]) / (
                    z_sorted[:, -1] - (z_sorted[:, -3] + z_sorted[:, -4]) / 2 + 1e-12)

        def untargeted_fn(z, labels):
            z_sorted = z.sort(axis=1)
            ind_sorted = z.argsort(axis=1)
            ind = (ind_sorted[:, -1] == labels).float32()
            return -(z[indices, labels] - z_sorted[:, -2] * ind - z_sorted[:, -1] * (1. - ind)) / (
                    z_sorted[:, -1] - z_sorted[:, -3] + 1e-12)

        return get_modified_loss(model, inputs, labels, untargeted_fn, targeted_fn, targeted, modifier)

    return loss_fn


def get_logit_loss_fn(model: Model, labels: ep.Tensor, targeted: bool, modifier: dict,
                   ) -> Callable[[ep.Tensor, ep.Tensor], ep.Tensor]:
    """
    modifier needs to contain 'match_target' tensor with dimension K x K
    """

    def get_match_target(targets, labels):
        import torch
        result = []
        if isinstance(labels, ep.Tensor):
            labels = labels.raw
        for label in labels:
            result.append(targets[label, :])
        return ep.astensor(torch.tensor(result).to("cuda"))

    def loss_fn(inputs: ep.Tensor) -> ep.Tensor:

        assert('match_target' in modifier)
        modifier['loss_diff'] = False
        def targeted_fn(z, labels):
            targets = get_match_target(modifier['match_target'], labels)
            return -(z - targets).square().sum(axis=1)

        def untargeted_fn(z, labels):
            targets = get_match_target(modifier['match_target'], labels)
            return (z - targets).square().sum(axis=1)

        return get_modified_loss(model, inputs, labels, untargeted_fn, targeted_fn, targeted, modifier)

    return loss_fn


def extract_target_logits(model: Model, inputs: ep.Tensor, labels: ep.Tensor):
    """
    This implementation uses any correctly classified sample as the target logit
    """
    if not isinstance(labels, ep.Tensor):
        labels, _ = ep.astensor_(labels)
    num_classes = 10  # Hack for CIFAR10
    result = np.zeros([num_classes, num_classes])
    present = np.zeros(num_classes)
    Z = model(inputs)  # all the logits

    if isinstance(Z, ep.Tensor):
        Z = Z.raw
    else:
        Z = Z.detach()

    for i in range(labels.shape[0]):
        t = labels[i].raw.item()
        if present[t]:
            continue
        z = Z[i:i+1]
        if z.argmax() == t:
            result[t, :] = z.cpu()
            present[t] = 1
        if sum(present) == num_classes:
            break

    return result
