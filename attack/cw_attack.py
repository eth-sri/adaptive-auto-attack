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
This module implements the L2 and LInf optimized attacks `CarliniL2Method` and `CarliniLInfMethod` of Carlini and Wagner
(2016). These attacks are among the most effective white-box attacks and should be used among the primary attacks to
evaluate potential defences. A major difference with respect to the original implementation
(https://github.com/carlini/nn_robust_attacks) is that this implementation uses line search in the optimization of the
attack objective.

| Paper link: https://arxiv.org/abs/1608.04644
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.classifiers.classifier import ClassifierGradients
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, get_labels_np_array, tanh_to_original, original_to_tanh
from art.utils import check_and_transform_label_format
from art.exceptions import ClassifierError

logger = logging.getLogger(__name__)


class CarliniL2Method(EvasionAttack):
    """
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is among the most effective and should be used
    among the primary attacks to evaluate potential defences. A major difference wrt to the original implementation
    (https://github.com/carlini/nn_robust_attacks) is that we use line search in the optimization of the attack
    objective.

    | Paper link: https://arxiv.org/abs/1608.04644
    """

    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "binary_search_steps",
        "initial_const",
        "max_halving",
        "max_doubling",
        "batch_size",
    ]

    def __init__(
        self,
        classifier,
        confidence=0.0,
        targeted=False,
        learning_rate=0.01,
        binary_search_steps=10,
        max_iter=10,
        initial_const=0.01,
        max_halving=5,
        max_doubling=5,
        batch_size=1,
    ):
        """
        Create a Carlini L_2 attack instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
                from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better results
                but are slower to converge.
        :type learning_rate: `float`
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value). If
                                    `binary_search_steps` is large, then the algorithm is not very sensitive to the
                                    value of `initial_const`. Note that the values gamma=0.999999 and c_upper=10e10 are
                                    hardcoded with the same values used by the authors of the method.
        :type binary_search_steps: `int`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance and
                confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
                Carlini and Wagner (2016).
        :type initial_const: `float`
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :type max_halving: `int`
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :type max_doubling: `int`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(CarliniL2Method, self).__init__(classifier)
        if not isinstance(classifier, ClassifierGradients):
            raise ClassifierError(self.__class__, [ClassifierGradients], classifier)

        kwargs = {
            "confidence": confidence,
            "targeted": targeted,
            "learning_rate": learning_rate,
            "binary_search_steps": binary_search_steps,
            "max_iter": max_iter,
            "initial_const": initial_const,
            "max_halving": max_halving,
            "max_doubling": max_doubling,
            "batch_size": batch_size,
        }
        assert self.set_params(**kwargs)

        # There are internal hyperparameters:
        # Abort binary search for c if it exceeds this threshold (suggested in Carlini and Wagner (2016)):
        self._c_upper_bound = 10e10

        # Smooth arguments of arctanh by multiplying with this constant to avoid division by zero.
        # It appears this is what Carlini and Wagner (2016) are alluding to in their footnote 8. However, it is not
        # clear how their proposed trick ("instead of scaling by 1/2 we scale by 1/2 + eps") works in detail.
        self._tanh_smoother = 0.999999

    def _loss(self, x, x_adv, target, c_weight):
        """
        Compute the objective function value.

        :param x: An array with the original input.
        :type x: `np.ndarray`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :param target: An array with the target class (one-hot encoded).
        :type target: `np.ndarray`
        :param c_weight: Weight of the loss term aiming for classification as target.
        :type c_weight: `float`
        :return: A tuple holding the current logits, l2 distance and overall loss.
        :rtype: `(float, float, float)`
        """
        l2dist = np.sum(np.square(x - x_adv).reshape(x.shape[0], -1), axis=1)
        z_predicted = self.classifier.predict(
            np.array(x_adv, dtype=ART_NUMPY_DTYPE), logits=True, batch_size=self.batch_size
        )
        z_target = np.sum(z_predicted * target, axis=1)
        z_other = np.max(z_predicted * (1 - target) + (np.min(z_predicted, axis=1) - 1)[:, np.newaxis] * target, axis=1)

        # The following differs from the exact definition given in Carlini and Wagner (2016). There (page 9, left
        # column, last equation), the maximum is taken over Z_other - Z_target (or Z_target - Z_other respectively)
        # and -confidence. However, it doesn't seem that that would have the desired effect (loss term is <= 0 if and
        # only if the difference between the logit of the target and any other class differs by at least confidence).
        # Hence the rearrangement here.

        if self.targeted:
            # if targeted, optimize for making the target class most likely
            loss = np.maximum(z_other - z_target + self.confidence, np.zeros(x.shape[0]))
        else:
            # if untargeted, optimize for making any other class most likely
            loss = np.maximum(z_target - z_other + self.confidence, np.zeros(x.shape[0]))

        return z_predicted, l2dist, c_weight * loss + l2dist

    def _loss_gradient(self, z_logits, target, x, x_adv, x_adv_tanh, c_weight, clip_min, clip_max):
        """
        Compute the gradient of the loss function.

        :param z_logits: An array with the current logits.
        :type z_logits: `np.ndarray`
        :param target: An array with the target class (one-hot encoded).
        :type target: `np.ndarray`
        :param x: An array with the original input.
        :type x: `np.ndarray`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :param x_adv_tanh: An array with the adversarial input in tanh space.
        :type x_adv_tanh: `np.ndarray`
        :param c_weight: Weight of the loss term aiming for classification as target.
        :type c_weight: `float`
        :param clip_min: Minimum clipping value.
        :type clip_min: `float`
        :param clip_max: Maximum clipping value.
        :type clip_max: `float`
        :return: An array with the gradient of the loss function.
        :type target: `np.ndarray`
        """
        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)

        loss_gradient = self.classifier.class_gradient(x_adv, label=i_add)
        loss_gradient -= self.classifier.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x.shape)

        c_mult = c_weight
        for _ in range(len(x.shape) - 1):
            c_mult = c_mult[:, np.newaxis]

        loss_gradient *= c_mult
        loss_gradient += 2 * (x_adv - x)
        loss_gradient *= clip_max - clip_min
        loss_gradient *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)

        return loss_gradient

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. If `self.targeted`
                  is true, then `y_val` represents the target labels. Otherwise, the targets are the original class
                  labels.
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.classifier.nb_classes())
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if hasattr(self.classifier, "clip_values") and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
        else:
            clip_min, clip_max = np.amin(x), np.amax(x)

        # Assert that, if attack is targeted, y_val is provided:
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in range(nb_batches):
            logger.debug("Processing batch %i out of %i", batch_id, nb_batches)

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]

            # The optimization is performed in tanh space to keep the adversarial images bounded in correct range
            x_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)

            # Initialize binary search:
            c_current = self.initial_const * np.ones(x_batch.shape[0])
            c_lower_bound = np.zeros(x_batch.shape[0])
            c_double = np.ones(x_batch.shape[0]) > 0

            # Initialize placeholders for best l2 distance and attack found so far
            best_l2dist = np.inf * np.ones(x_batch.shape[0])
            best_x_adv_batch = x_batch.copy()

            for bss in range(self.binary_search_steps):
                logger.debug(
                    "Binary search step %i out of %i (c_mean==%f)", bss, self.binary_search_steps, np.mean(c_current)
                )
                nb_active = int(np.sum(c_current < self._c_upper_bound))
                logger.debug(
                    "Number of samples with c_current < _c_upper_bound: %i out of %i", nb_active, x_batch.shape[0]
                )
                if nb_active == 0:
                    break
                learning_rate = self.learning_rate * np.ones(x_batch.shape[0])

                # Initialize perturbation in tanh space:
                x_adv_batch = x_batch.copy()
                x_adv_batch_tanh = x_batch_tanh.copy()

                z_logits, l2dist, loss = self._loss(x_batch, x_adv_batch, y_batch, c_current)
                attack_success = loss - l2dist <= 0
                overall_attack_success = attack_success

                for i_iter in range(self.max_iter):
                    logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)
                    logger.debug("Average Loss: %f", np.mean(loss))
                    logger.debug("Average L2Dist: %f", np.mean(l2dist))
                    logger.debug("Average Margin Loss: %f", np.mean(loss - l2dist))
                    logger.debug(
                        "Current number of succeeded attacks: %i out of %i",
                        int(np.sum(attack_success)),
                        len(attack_success),
                    )

                    improved_adv = attack_success & (l2dist < best_l2dist)
                    logger.debug("Number of improved L2 distances: %i", int(np.sum(improved_adv)))
                    if np.sum(improved_adv) > 0:
                        best_l2dist[improved_adv] = l2dist[improved_adv]
                        best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                    active = (c_current < self._c_upper_bound) & (learning_rate > 0)
                    nb_active = int(np.sum(active))
                    logger.debug(
                        "Number of samples with c_current < _c_upper_bound and learning_rate > 0: %i out of %i",
                        nb_active,
                        x_batch.shape[0],
                    )
                    if nb_active == 0:
                        break

                    # compute gradient:
                    logger.debug("Compute loss gradient")
                    perturbation_tanh = -self._loss_gradient(
                        z_logits[active],
                        y_batch[active],
                        x_batch[active],
                        x_adv_batch[active],
                        x_adv_batch_tanh[active],
                        c_current[active],
                        clip_min,
                        clip_max,
                    )

                    # perform line search to optimize perturbation
                    # first, halve the learning rate until perturbation actually decreases the loss:
                    prev_loss = loss.copy()
                    best_loss = loss.copy()
                    best_lr = np.zeros(x_batch.shape[0])
                    halving = np.zeros(x_batch.shape[0])

                    for i_halve in range(self.max_halving):
                        logger.debug("Perform halving iteration %i out of %i", i_halve, self.max_halving)
                        do_halving = loss[active] >= prev_loss[active]
                        logger.debug("Halving to be performed on %i samples", int(np.sum(do_halving)))
                        if np.sum(do_halving) == 0:
                            break
                        active_and_do_halving = active.copy()
                        active_and_do_halving[active] = do_halving

                        lr_mult = learning_rate[active_and_do_halving]
                        for _ in range(len(x.shape) - 1):
                            lr_mult = lr_mult[:, np.newaxis]

                        x_adv1 = x_adv_batch_tanh[active_and_do_halving]
                        new_x_adv_batch_tanh = x_adv1 + lr_mult * perturbation_tanh[do_halving]
                        new_x_adv_batch = tanh_to_original(
                            new_x_adv_batch_tanh, clip_min, clip_max, self._tanh_smoother
                        )
                        _, l2dist[active_and_do_halving], loss[active_and_do_halving] = self._loss(
                            x_batch[active_and_do_halving],
                            new_x_adv_batch,
                            y_batch[active_and_do_halving],
                            c_current[active_and_do_halving],
                        )

                        logger.debug("New Average Loss: %f", np.mean(loss))
                        logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                        logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))

                        best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]
                        learning_rate[active_and_do_halving] /= 2
                        halving[active_and_do_halving] += 1
                    learning_rate[active] *= 2

                    # if no halving was actually required, double the learning rate as long as this
                    # decreases the loss:
                    for i_double in range(self.max_doubling):
                        logger.debug("Perform doubling iteration %i out of %i", i_double, self.max_doubling)
                        do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                        logger.debug("Doubling to be performed on %i samples", int(np.sum(do_doubling)))
                        if np.sum(do_doubling) == 0:
                            break
                        active_and_do_doubling = active.copy()
                        active_and_do_doubling[active] = do_doubling
                        learning_rate[active_and_do_doubling] *= 2

                        lr_mult = learning_rate[active_and_do_doubling]
                        for _ in range(len(x.shape) - 1):
                            lr_mult = lr_mult[:, np.newaxis]

                        x_adv2 = x_adv_batch_tanh[active_and_do_doubling]
                        new_x_adv_batch_tanh = x_adv2 + lr_mult * perturbation_tanh[do_doubling]
                        new_x_adv_batch = tanh_to_original(
                            new_x_adv_batch_tanh, clip_min, clip_max, self._tanh_smoother
                        )
                        _, l2dist[active_and_do_doubling], loss[active_and_do_doubling] = self._loss(
                            x_batch[active_and_do_doubling],
                            new_x_adv_batch,
                            y_batch[active_and_do_doubling],
                            c_current[active_and_do_doubling],
                        )
                        logger.debug("New Average Loss: %f", np.mean(loss))
                        logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                        logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))
                        best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]

                    learning_rate[halving == 1] /= 2

                    update_adv = best_lr[active] > 0
                    logger.debug("Number of adversarial samples to be finally updated: %i", int(np.sum(update_adv)))

                    if np.sum(update_adv) > 0:
                        active_and_update_adv = active.copy()
                        active_and_update_adv[active] = update_adv
                        best_lr_mult = best_lr[active_and_update_adv]
                        for _ in range(len(x.shape) - 1):
                            best_lr_mult = best_lr_mult[:, np.newaxis]

                        x_adv4 = x_adv_batch_tanh[active_and_update_adv]
                        best_lr1 = best_lr_mult * perturbation_tanh[update_adv]
                        x_adv_batch_tanh[active_and_update_adv] = x_adv4 + best_lr1

                        x_adv6 = x_adv_batch_tanh[active_and_update_adv]
                        x_adv_batch[active_and_update_adv] = tanh_to_original(
                            x_adv6, clip_min, clip_max, self._tanh_smoother
                        )
                        (
                            z_logits[active_and_update_adv],
                            l2dist[active_and_update_adv],
                            loss[active_and_update_adv],
                        ) = self._loss(
                            x_batch[active_and_update_adv],
                            x_adv_batch[active_and_update_adv],
                            y_batch[active_and_update_adv],
                            c_current[active_and_update_adv],
                        )
                        attack_success = loss - l2dist <= 0
                        overall_attack_success = overall_attack_success | attack_success

                # Update depending on attack success:
                improved_adv = attack_success & (l2dist < best_l2dist)
                logger.debug("Number of improved L2 distances: %i", int(np.sum(improved_adv)))

                if np.sum(improved_adv) > 0:
                    best_l2dist[improved_adv] = l2dist[improved_adv]
                    best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                c_double[overall_attack_success] = False
                c_current[overall_attack_success] = (c_lower_bound + c_current)[overall_attack_success] / 2

                c_old = c_current
                c_current[~overall_attack_success & c_double] *= 2

                c_current1 = (c_current - c_lower_bound)[~overall_attack_success & ~c_double]
                c_current[~overall_attack_success & ~c_double] += c_current1 / 2
                c_lower_bound[~overall_attack_success] = c_old[~overall_attack_success]

            x_adv[batch_index_1:batch_index_2] = best_x_adv_batch

        logger.info(
            "Success rate of C&W L_2 attack: %.2f%%",
            100 * compute_success(self.classifier, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
               from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :type learning_rate: `float`
        :param binary_search_steps: number of times to adjust constant with binary search (positive value)
        :type binary_search_steps: `int`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
               importance of distance and confidence. If binary_search_steps is large,
               the initial constant is not important. The default value 1e-4 is suggested in Carlini and Wagner (2016).
        :type initial_const: `float`
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :type max_halving: `int`
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :type max_doubling: `int`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(CarliniL2Method, self).set_params(**kwargs)

        if not isinstance(self.binary_search_steps, (int, np.int)) or self.binary_search_steps < 0:
            raise ValueError("The number of binary search steps must be a non-negative integer.")

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.max_halving, (int, np.int)) or self.max_halving < 1:
            raise ValueError("The number of halving steps must be an integer greater than zero.")

        if not isinstance(self.max_doubling, (int, np.int)) or self.max_doubling < 1:
            raise ValueError("The number of doubling steps must be an integer greater than zero.")

        if not isinstance(self.batch_size, (int, np.int)) or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")

        return True


class CarliniLInfMethod(EvasionAttack):
    """
    This is a modified version of the L_2 optimized attack of Carlini and Wagner (2016). It controls the L_Inf
    norm, i.e. the maximum perturbation applied to each pixel.
    """

    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "max_halving",
        "max_doubling",
        "eps",
        "batch_size",
    ]

    def __init__(
        self,
        classifier,
        aux_classifier = None,
        confidence=0.0,
        targeted=False,
        learning_rate=0.01,
        max_iter=10,
        max_halving=5,
        max_doubling=5,
        eps=0.3,
        batch_size=128,
    ):
        """
        Create a Carlini L_Inf attack instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
                from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
                results but are slower to converge.
        :type learning_rate: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :type max_halving: `int`
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :type max_doubling: `int`
        :param eps: An upper bound for the L_0 norm of the adversarial perturbation.
        :type eps: `float`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        :param expectation: An expectation over transformations to be applied when computing
                            classifier gradients and predictions.
        :type expectation: :class:`.ExpectationOverTransformations`
        """
        super(CarliniLInfMethod, self).__init__(classifier)
        if not isinstance(classifier, ClassifierGradients):
            raise ClassifierError(self.__class__, [ClassifierGradients], classifier)

        kwargs = {
            "confidence": confidence,
            "targeted": targeted,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_halving": max_halving,
            "max_doubling": max_doubling,
            "eps": eps,
            "batch_size": batch_size,
        }
        assert self.set_params(**kwargs)

        # There is one internal hyperparameter:
        # Smooth arguments of arctanh by multiplying with this constant to avoid division by zero:
        self._tanh_smoother = 0.999999

    def _loss(self, x_adv, target):
        """
        Compute the objective function value.

        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :param target: An array with the target class (one-hot encoded).
        :type target: `np.ndarray`
        :return: A tuple holding the current predictions and overall loss.
        :rtype: `(float, float)`
        """
        z_predicted = self.classifier.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE), batch_size=self.batch_size)
        z_target = np.sum(z_predicted * target, axis=1)
        z_other = np.max(z_predicted * (1 - target) + (np.min(z_predicted, axis=1) - 1)[:, np.newaxis] * target, axis=1)

        if self.targeted:
            # if targeted, optimize for making the target class most likely
            loss = np.maximum(z_other - z_target + self.confidence, np.zeros(x_adv.shape[0]))
        else:
            # if untargeted, optimize for making any other class most likely
            loss = np.maximum(z_target - z_other + self.confidence, np.zeros(x_adv.shape[0]))

        # loss_fn = get_loss_fn(model=self.aux_classifier, labels=self.targeted, type='hinge', targeted=True,
        #             modifier={'indiv': 1, 'logits': 1}, param={'softmax': False})
        # _, loss, z_predicted = loss_fn(x_adv)

        return z_predicted, loss

    def _loss_gradient(self, z_logits, target, x_adv, x_adv_tanh, clip_min, clip_max):  # lgtm [py/similar-function]
        """
        Compute the gradient of the loss function.

        :param z_logits: An array with the current predictions.
        :type z_logits: `np.ndarray`
        :param target: An array with the target class (one-hot encoded).
        :type target: `np.ndarray`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :param x_adv_tanh: An array with the adversarial input in tanh space.
        :type x_adv_tanh: `np.ndarray`
        :param clip_min: Minimum clipping values.
        :type clip_min: `np.ndarray`
        :param clip_max: Maximum clipping values.
        :type clip_max: `np.ndarray`
        :return: An array with the gradient of the loss function.
        :type target: `np.ndarray`
        """
        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)

        loss_gradient = self.classifier.class_gradient(x_adv, label=i_add)
        loss_gradient -= self.classifier.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x_adv.shape)

        # loss_fn = get_loss_fn(model=self.estimator, type='hinge', labels=target, targeted=True, modifier={}, param={})
        # grad = ep.value_and_grad_fn(x_adv, loss_fn)
        # grad *= clip_max - clip_min

        loss_gradient *= clip_max - clip_min
        loss_gradient *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)

        return loss_gradient

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y_val` represents the target labels. Otherwise, the
                  targets are the original class labels.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.classifier.nb_classes())
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if hasattr(self.classifier, "clip_values") and self.classifier.clip_values is not None:
            clip_min_per_pixel, clip_max_per_pixel = self.classifier.clip_values
        else:
            clip_min_per_pixel, clip_max_per_pixel = np.amin(x), np.amax(x)

        # Assert that, if attack is targeted, y_val is provided:
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in range(nb_batches):
            logger.debug("Processing batch %i out of %i", batch_id, nb_batches)

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]

            # Determine values for later clipping
            clip_min = np.clip(x_batch - self.eps, clip_min_per_pixel, clip_max_per_pixel)
            clip_max = np.clip(x_batch + self.eps, clip_min_per_pixel, clip_max_per_pixel)

            # The optimization is performed in tanh space to keep the
            # adversarial images bounded from clip_min and clip_max.
            x_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)

            # Initialize perturbation in tanh space:
            x_adv_batch = x_batch.copy()
            x_adv_batch_tanh = x_batch_tanh.copy()

            # Initialize optimization:
            z_logits, loss = self._loss(x_adv_batch, y_batch)
            attack_success = loss <= 0
            learning_rate = self.learning_rate * np.ones(x_batch.shape[0])

            for i_iter in range(self.max_iter):
                logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)
                logger.debug("Average Loss: %f", np.mean(loss))

                logger.debug("Successful attack samples: %i out of %i", int(np.sum(attack_success)), x_batch.shape[0])

                # only continue optimization for those samples where attack hasn't succeeded yet:
                active = ~attack_success
                if np.sum(active) == 0:
                    break

                # compute gradient:
                logger.debug("Compute loss gradient")
                perturbation_tanh = -self._loss_gradient(
                    z_logits[active],
                    y_batch[active],
                    x_adv_batch[active],
                    x_adv_batch_tanh[active],
                    clip_min[active],
                    clip_max[active],
                )

                # perform line search to optimize perturbation
                # first, halve the learning rate until perturbation actually decreases the loss:
                prev_loss = loss.copy()
                best_loss = loss.copy()
                best_lr = np.zeros(x_batch.shape[0])
                halving = np.zeros(x_batch.shape[0])

                for i_halve in range(self.max_halving):
                    logger.debug("Perform halving iteration %i out of %i", i_halve, self.max_halving)
                    do_halving = loss[active] >= prev_loss[active]
                    logger.debug("Halving to be performed on %i samples", int(np.sum(do_halving)))
                    if np.sum(do_halving) == 0:
                        break
                    active_and_do_halving = active.copy()
                    active_and_do_halving[active] = do_halving

                    lr_mult = learning_rate[active_and_do_halving]
                    for _ in range(len(x.shape) - 1):
                        lr_mult = lr_mult[:, np.newaxis]

                    adv_10 = x_adv_batch_tanh[active_and_do_halving]
                    new_x_adv_batch_tanh = adv_10 + lr_mult * perturbation_tanh[do_halving]

                    new_x_adv_batch = tanh_to_original(
                        new_x_adv_batch_tanh, clip_min[active_and_do_halving], clip_max[active_and_do_halving]
                    )
                    _, loss[active_and_do_halving] = self._loss(new_x_adv_batch, y_batch[active_and_do_halving])
                    logger.debug("New Average Loss: %f", np.mean(loss))
                    logger.debug("Loss: %s", str(loss))
                    logger.debug("Prev_loss: %s", str(prev_loss))
                    logger.debug("Best_loss: %s", str(best_loss))

                    best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                    best_loss[loss < best_loss] = loss[loss < best_loss]
                    learning_rate[active_and_do_halving] /= 2
                    halving[active_and_do_halving] += 1
                learning_rate[active] *= 2

                # if no halving was actually required, double the learning rate as long as this
                # decreases the loss:
                for i_double in range(self.max_doubling):
                    logger.debug("Perform doubling iteration %i out of %i", i_double, self.max_doubling)
                    do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                    logger.debug("Doubling to be performed on %i samples", int(np.sum(do_doubling)))
                    if np.sum(do_doubling) == 0:
                        break
                    active_and_do_doubling = active.copy()
                    active_and_do_doubling[active] = do_doubling
                    learning_rate[active_and_do_doubling] *= 2

                    lr_mult = learning_rate[active_and_do_doubling]
                    for _ in range(len(x.shape) - 1):
                        lr_mult = lr_mult[:, np.newaxis]

                    x_adv15 = x_adv_batch_tanh[active_and_do_doubling]
                    new_x_adv_batch_tanh = x_adv15 + lr_mult * perturbation_tanh[do_doubling]
                    new_x_adv_batch = tanh_to_original(
                        new_x_adv_batch_tanh, clip_min[active_and_do_doubling], clip_max[active_and_do_doubling]
                    )
                    _, loss[active_and_do_doubling] = self._loss(new_x_adv_batch, y_batch[active_and_do_doubling])
                    logger.debug("New Average Loss: %f", np.mean(loss))
                    best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                    best_loss[loss < best_loss] = loss[loss < best_loss]

                learning_rate[halving == 1] /= 2

                update_adv = best_lr[active] > 0
                logger.debug("Number of adversarial samples to be finally updated: %i", int(np.sum(update_adv)))

                if np.sum(update_adv) > 0:
                    active_and_update_adv = active.copy()
                    active_and_update_adv[active] = update_adv
                    best_lr_mult = best_lr[active_and_update_adv]
                    for _ in range(len(x.shape) - 1):
                        best_lr_mult = best_lr_mult[:, np.newaxis]

                    best_13 = best_lr_mult * perturbation_tanh[update_adv]
                    x_adv_batch_tanh[active_and_update_adv] = x_adv_batch_tanh[active_and_update_adv] + best_13
                    x_adv_batch[active_and_update_adv] = tanh_to_original(
                        x_adv_batch_tanh[active_and_update_adv],
                        clip_min[active_and_update_adv],
                        clip_max[active_and_update_adv],
                    )
                    z_logits[active_and_update_adv], loss[active_and_update_adv] = self._loss(
                        x_adv_batch[active_and_update_adv], y_batch[active_and_update_adv]
                    )
                    attack_success = loss <= 0

            # Update depending on attack success:
            x_adv_batch[~attack_success] = x_batch[~attack_success]
            x_adv[batch_index_1:batch_index_2] = x_adv_batch

        logger.info(
            "Success rate of C&W L_inf attack: %.2f%%",
            100 * compute_success(self.classifier, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
               from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :type learning_rate: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :type max_halving: `int`
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :type max_doubling: `int`
        :param eps: An upper bound for the L_0 norm of the adversarial perturbation.
        :type eps: `float`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(CarliniLInfMethod, self).set_params(**kwargs)

        if self.eps <= 0:
            raise ValueError("The eps parameter must be strictly positive.")

        if not isinstance(self.max_iter, (int, np.int, np.int64)) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.max_halving, (int, np.int, np.int64)) or self.max_halving < 1:
            raise ValueError("The number of halving steps must be an integer greater than zero.")

        if not isinstance(self.max_doubling, (int, np.int, np.int64)) or self.max_doubling < 1:
            raise ValueError("The number of doubling steps must be an integer greater than zero.")

        if not isinstance(self.batch_size, (int, np.int)) or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")

        return True