# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the classifier `PyTorchClassifier` for PyTorch models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import torch
import torch.nn as nn
import numpy as np
# from tqdm import tqdm
from collections import OrderedDict
from art.utils import check_and_transform_label_format

from defenses.defense_module import DefenseModule
from utils.classifier import Classifier
from defenses.bpda import BPDAWrapper

logger = logging.getLogger(__name__)


class DefenseModel:
    def __init__(
        self,
        model,
        loss,
        optimizer,
        nb_classes,
        channel_index=1,
        detector=None,
        clip_values=None,
        pre_defences=None,
        post_defences=None,
        preprocessing=(0, 1),
        device="cuda",
    ):
        """
        Initialization specifically for the PyTorch-based implementation.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :type model: `torch.nn.Module`
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :type loss: `torch.nn.modules.loss._Loss`
        :param optimizer: The optimizer used to train the classifier.
        :type optimizer: `torch.optim.Optimizer`
        :param nb_classes: The number of classes of the model.
        :type nb_classes: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param pre_defences: Preprocessing defence(s) to be applied by the classifier.
        :type pre_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param post_defences: Postprocessing defence(s) to be applied by the classifier.
        :type post_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :type device_type: `string`
        """

        self._clip_values = clip_values
        if clip_values is not None:
            if len(clip_values) != 2:
                raise ValueError(
                    "`clip_values` should be a tuple of 2 floats or arrays containing the allowed data range."
                )
            if np.array(clip_values[0] >= clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")
        if isinstance(pre_defences, DefenseModule):
            self.pre_defences = [pre_defences]
        else:
            self.pre_defences = pre_defences
        if isinstance(post_defences, DefenseModule):
            self.post_defences = [post_defences]
        else:
            self.post_defences = post_defences

        if preprocessing is not None and len(preprocessing) != 2:
            raise ValueError(
                "`preprocessing` should be a tuple of 2 floats with the values to subtract and divide"
                "the model inputs."
            )
        # Set device
        if device == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(cuda_idx))

        self.preprocessing = preprocessing
        self.loss = loss
        self.optimizer = optimizer
        self.detector = detector
        self.train_model, self.classifier = self._get_models(model)
        self._channel_index = channel_index
        self._nb_classes = nb_classes
        self._learning_phase = None
        self.classifier.to(self.device)

        # Index of layer at which the class gradients should be calculated
        self._layer_idx_gradients = -1
        # Check if the loss function requires as input index labels instead of one-hot-encoded labels
        if isinstance(loss, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss)):
            self._reduce_labels = True
        else:
            self._reduce_labels = False

    def _get_models(self, model):
        dictforward = OrderedDict()
        dict_train = OrderedDict()
        if self.pre_defences is not None:
            for i, pre_defense in enumerate(self.pre_defences):
                if pre_defense.apply_fit:
                    pre_defense_wrap = BPDAWrapper(pre_defense) #, forwardsub=lambda x: x)
                    dict_train["pre_defense" + str(i)] = pre_defense_wrap
                if pre_defense.apply_predict:
                    pre_defense_wrap = BPDAWrapper(pre_defense) #, forwardsub=lambda x: x)
                    dictforward["pre_defense" + str(i)] = pre_defense_wrap
        dictforward["model"] = model
        dict_train["model"] = model
        if self.post_defences is not None:
            for i, post_defense in enumerate(self.post_defences):
                if post_defense.apply_fit:
                    dict_train["post_defense"+str(i)] = post_defense
                if post_defense.apply_predict:
                    dictforward["post_defense"+str(i)] = post_defense
        train_model = Classifier(nn.Sequential(dict_train))
        eval_model = Classifier(nn.Sequential(dictforward))
        return train_model.to(self.device), eval_model.to(self.device)

    def predict(self, x, batch_size=128, **kwargs):
        if self.detector is not None:
            det = torch.argmax(self.detector(x), dim=1).to(torch.bool).to(self.device)
        else:
            # return all True in the absence of a detector
            det = torch.ones(x.shape[0]).to(torch.bool).to(self.device)
        x = x.to(self.device)
        return [self.classifier(x), det]

    def detectorfit(self, loader, nb_epochs=10):
        self.detector.fit(loader, self.loss, self.optimizer, nb_epochs=nb_epochs)

    def fit(self, loader, nb_epochs=10):
        self.train_model.fit(loader, self.loss, self.optimizer, nb_epochs=nb_epochs)

    def advfit(self, loader, attack, epsilon, nb_epochs=10, ratio=0.5, **kwargs):
        self.train_model.advfit(loader, self.loss, self.optimizer, attack=attack, epsilon=epsilon,
                                nb_epochs=nb_epochs, ratio=ratio, **kwargs)

    def _apply_preprocessing(self, x, y, fit):
        """
        Apply all defences and preprocessing operations on the inputs `(x, y)`. This function has to be applied to all
        raw inputs (x, y) provided to the classifier.

        :param x: Features, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :param y: Target values (class labels), where first dimension is the number of samples.
        :type y: `np.ndarray` or `None`
        :param fit: `True` if the defences are applied during training.
        :type fit: `bool`
        :return: Value of the data after applying the defences.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.nb_classes())
        x_preprocessed, y_preprocessed = self._apply_preprocessing_defences(x, y, fit=fit)
        x_preprocessed = self._apply_preprocessing_standardisation(x_preprocessed)
        return x_preprocessed, y_preprocessed

    def _apply_preprocessing_standardisation(self, x):
        """
        Apply standardisation to input data `x`.

        :param x: Input data, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :return: Array for `x` with the standardized data.
        :rtype: `np.ndarray`
        :raises: `TypeError`
        """
        if x.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            raise TypeError(
                "The data type of input data `x` is {} and cannot represent negative values. Consider "
                "changing the data type of the input data `x` to a type that supports negative values e.g. "
                "np.float32.".format(x.dtype)
            )
        if self.preprocessing is not None:
            sub, div = self.preprocessing
            sub = torch.tensor(sub).to(self._device, dtype=torch.float32)
            div = torch.tensor(div).to(self._device, dtype=torch.float32)
            res = x - sub
            res = res / div
        else:
            res = x
        return res

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        import torch

        if not (
            (label is None)
            or (isinstance(label, (int, np.integer)) and label in range(self._nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self._nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % label)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
        x_preprocessed = torch.from_numpy(x_preprocessed).to(self._device)

        # Compute gradients
        if self._layer_idx_gradients < 0:
            x_preprocessed.requires_grad = True

        # Run prediction
        model_outputs = self._model(x_preprocessed)

        # Set where to get gradient
        if self._layer_idx_gradients >= 0:
            input_grad = model_outputs[self._layer_idx_gradients]
        else:
            input_grad = x_preprocessed

        # Set where to get gradient from
        preds = model_outputs[-1]

        # Compute the gradient
        grads = []

        def save_grad():
            def hook(grad):
                grads.append(grad.cpu().numpy().copy())
                grad.data.zero_()

            return hook

        input_grad.register_hook(save_grad())

        self._model.zero_grad()
        if label is None:
            for i in range(self.nb_classes()):
                torch.autograd.backward(
                    preds[:, i], torch.Tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True
                )

        elif isinstance(label, (int, np.integer)):
            torch.autograd.backward(
                preds[:, label], torch.Tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True
            )
        else:
            unique_label = list(np.unique(label))
            for i in unique_label:
                torch.autograd.backward(
                    preds[:, i], torch.Tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True
                )

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]

        grads = np.swapaxes(np.array(grads), 0, 1)
        grads = self._apply_preprocessing_gradient(x, grads)

        return grads

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        import torch

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
        inputs_t.requires_grad = True

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(y_preprocessed).to(self._device)

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        loss = self.loss(model_outputs[-1], labels_t)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = inputs_t.grad.cpu().numpy().copy()
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x.shape

        return grads

    # @property
    # def layer_names(self):
    #     """
    #     Return the hidden layers in the model, if applicable.
    #
    #     :return: The hidden layers in the model, input and output layers excluded.
    #     :rtype: `list`
    #
    #     .. warning:: `layer_names` tries to infer the internal structure of the model.
    #                  This feature comes with no guarantees on the correctness of the result.
    #                  The intended order of the layers tries to match their order in the model, but this is not
    #                  guaranteed either. In addition, the function can only infer the internal layers if the input
    #                  model is of type `nn.Sequential`, otherwise, it will only return the logit layer.
    #     """
    #     return self._layer_names

    # def get_activations(self, x, layer, batch_size=128):
    #     """
    #     Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
    #     `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
    #     calling `layer_names`.
    #
    #     :param x: Input for computing the activations.
    #     :type x: `np.ndarray`
    #     :param layer: Layer for computing the activations
    #     :type layer: `int` or `str`
    #     :param batch_size: Size of batches.
    #     :type batch_size: `int`
    #     :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
    #     :rtype: `np.ndarray`
    #     """
    #     import torch
    #
    #     # Apply defences
    #     x_preprocessed, _ = self._apply_preprocessing(x=x, y=None, fit=False)
    #
    #     # Get index of the extracted layer
    #     if isinstance(layer, six.string_types):
    #         if layer not in self._layer_names:
    #             raise ValueError("Layer name %s not supported" % layer)
    #         layer_index = self._layer_names.index(layer)
    #
    #     elif isinstance(layer, (int, np.integer)):
    #         layer_index = layer
    #
    #     else:
    #         raise TypeError("Layer must be of type str or int")
    #
    #     # Run prediction with batch processing
    #     results = []
    #     num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
    #     for m in range(num_batch):
    #         # Batch indexes
    #         begin, end = m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0])
    #
    #         # Run prediction for the current batch
    #         layer_output = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))[layer_index]
    #         results.append(layer_output.detach().cpu().numpy())
    #
    #     results = np.concatenate(results)
    #
    #     return results

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        if isinstance(train, bool):
            self._learning_phase = train
            self._model.training(train)

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    def __getstate__(self):
        """
        Use to ensure `PytorchClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        :rtype: `dict`
        """
        import time
        import copy

        # pylint: disable=W0212
        # disable pylint because access to _model required
        state = self.__dict__.copy()
        state["inner_model"] = copy.copy(state["_model"]._model)

        # Remove the unpicklable entries
        del state["_model_wrapper"]
        del state["_device"]
        del state["_model"]

        model_name = str(time.time())
        state["model_name"] = model_name
        self.save(model_name)

        return state

    def __setstate__(self, state):
        """
        Use to ensure `PytorchClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        :type state: `dict`
        """
        self.__dict__.update(state)

        # Load and update all functionality related to Pytorch
        import os
        import torch
        from art.config import ART_DATA_PATH

        # Recover model
        full_path = os.path.join(ART_DATA_PATH, state["model_name"])
        model = state["inner_model"]
        model.load_state_dict(torch.load(str(full_path) + ".model"))
        model.eval()
        self._model = self._make_model_wrapper(model)

        # Recover device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        # Recover optimizer
        self.optimizer.load_state_dict(torch.load(str(full_path) + ".optimizer"))

        self.__dict__.pop("model_name", None)
        self.__dict__.pop("inner_model", None)

    def __repr__(self):
        repr_ = (
            "%s(model=%r, loss=%r, optimizer=%r, nb_classes=%r, channel_index=%r, "
            "clip_values=%r, preprocessing_defences=%r, postprocessing_defences=%r, preprocessing=%r)"
            % (
                self.__module__ + "." + self.__class__.__name__,
                self._model,
                self.loss,
                self.optimizer,
                self.nb_classes(),
                self.channel_index,
                self.clip_values,
                self.pre_defences,
                self.post_defences,
                self.preprocessing,
            )
        )

        return repr_
