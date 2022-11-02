#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /optimization.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 24th 2022 01:39:25 pm                                                #
# Modified   : Monday October 24th 2022 01:57:41 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Optimization Configuration Definition Module

This module has been adapted from the TensorFlow / Models library and is subject to the terms
and conditions of the Apache License Version 2.0, included in the LICENSE document of this
package by reference.

Reference: :ref:`TensorFlow <https://github.com/tensorflow/models>`

"""
from typing import Optional, List
from dataclasses import dataclass

from csf.config.base import Config

# ------------------------------------------------------------------------------------------------ #


@dataclass
class BaseOptimizerConfig(Config):
    """Base optimizer config.

    Args:
        clipnorm: float >= 0 or None. If not None, Gradients will be clipped when
            their L2 norm exceeds this value.
        clipvalue: float >= 0 or None. If not None, Gradients will be clipped when
            their absolute value exceeds this value.
        global_clipnorm: float >= 0 or None. If not None, gradient of all weights is
            clipped so that their global norm is no higher than this value
    """

    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None
    global_clipnorm: Optional[float] = None


@dataclass
class SGDConfig(BaseOptimizerConfig):
    """Configuration for SGD optimizer.
    The attributes for this class matches the arguments of tf.keras.optimizer.SGD.

    Args:
        name: name of the optimizer.
        decay: decay rate for SGD optimizer.
        nesterov: nesterov for SGD optimizer.
        momentum: momentum for SGD optimizer.
    """

    name: str = "SGD"
    decay: float = 0.0
    nesterov: bool = False
    momentum: float = 0.0


@dataclass
class RMSPropConfig(BaseOptimizerConfig):
    """Configuration for RMSProp optimizer.
    The attributes for this class matches the arguments of
    tf.keras.optimizers.RMSprop.

    Args:
        name: name of the optimizer.
        rho: discounting factor for RMSprop optimizer.
        momentum: momentum for RMSprop optimizer.
        epsilon: epsilon value for RMSprop optimizer, help with numerical stability.
        centered: Whether to normalize gradients or not.
    """

    name: str = "RMSprop"
    rho: float = 0.9
    momentum: float = 0.0
    epsilon: float = 1e-7
    centered: bool = False


@dataclass
class AdagradConfig(BaseOptimizerConfig):
    """Configuration for Adagrad optimizer.
    The attributes of this class match the arguments of
    tf.keras.optimizer.Adagrad.

    Args:
        name: name of the optimizer.
        initial_accumulator_value: A floating point value. Starting value for the
            accumulators, must be non-negative.
        epsilon: A small floating point value to avoid zero denominator.
    """

    name: str = "Adagrad"
    initial_accumulator_value: float = 0.1
    epsilon: float = 1e-07


@dataclass
class AdamConfig(BaseOptimizerConfig):
    """Configuration for Adam optimizer.
    The attributes for this class matches the arguments of
    tf.keras.optimizer.Adam.

    Args:
        name: name of the optimizer.
        beta_1: decay rate for 1st order moments.
        beta_2: decay rate for 2st order moments.
        epsilon: epsilon value used for numerical stability in Adam optimizer.
        amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond".
    """

    name: str = "Adam"
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False


@dataclass
class AdamExperimentalConfig(BaseOptimizerConfig):
    """Configuration for experimental Adam optimizer.
    The attributes for this class matches the arguments of
    `tf.keras.optimizer.experimental.Adam`.

    Args:
        name: name of the optimizer.
        beta_1: decay rate for 1st order moments.
        beta_2: decay rate for 2st order moments.
        epsilon: epsilon value used for numerical stability in Adam optimizer.
        amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond".
        jit_compile: if True, jit compile will be used.
    """

    name: str = "Adam"
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False
    jit_compile: bool = False


@dataclass
class AdamWeightDecayConfig(BaseOptimizerConfig):
    """Configuration for Adam optimizer with weight decay.

    Args:
        name: name of the optimizer.
        beta_1: decay rate for 1st order moments.
        beta_2: decay rate for 2st order moments.
        epsilon: epsilon value used for numerical stability in the optimizer.
        amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond".
        weight_decay_rate: float. Weight decay rate. Default to 0.
        include_in_weight_decay: list[str], or None. List of weight names to include
        in weight decay.
        exclude_from_weight_decay: list[str], or None. List of weight names to not
        include in weight decay.
        gradient_clip_norm: A positive float. Clips the gradients to this maximum
        L2-norm. Default to 1.0.
    """

    name: str = "AdamWeightDecay"
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False
    weight_decay_rate: float = 0.0
    include_in_weight_decay: Optional[List[str]] = None
    exclude_from_weight_decay: Optional[List[str]] = None
    gradient_clip_norm: float = 1.0


@dataclass
class LAMBConfig(BaseOptimizerConfig):
    """Configuration for LAMB optimizer.
    The attributes for this class matches the arguments of
    tensorflow_addons.optimizers.LAMB.

    Args:
        name: name of the optimizer.
        beta_1: decay rate for 1st order moments.
        beta_2: decay rate for 2st order moments.
        epsilon: epsilon value used for numerical stability in LAMB optimizer.
        weight_decay_rate: float. Weight decay rate. Default to 0.
        exclude_from_weight_decay: List of regex patterns of variables excluded from
        weight decay. Variables whose name contain a substring matching the
        pattern will be excluded.
        exclude_from_layer_adaptation: List of regex patterns of variables excluded
        from layer adaptation. Variables whose name contain a substring matching
        the pattern will be excluded.
    """

    name: str = "LAMB"
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-6
    weight_decay_rate: float = 0.0
    exclude_from_weight_decay: Optional[List[str]] = None
    exclude_from_layer_adaptation: Optional[List[str]] = None


@dataclass
class EMAConfig(BaseOptimizerConfig):
    """Exponential moving average optimizer config.

    Args:
        name: 'str', name of the optimizer.
        trainable_weights_only: 'bool', if True, only model trainable weights will
        be updated. Otherwise, all model weights will be updated. This mainly
        affects batch normalization parameters.
        average_decay: 'float', average decay value.
        start_step: 'int', start step to apply moving average.
        dynamic_decay: 'bool', whether to apply dynamic decay or not.
    """

    name: str = "ExponentialMovingAverage"
    trainable_weights_only: bool = True
    average_decay: float = 0.99
    start_step: int = 0
    dynamic_decay: bool = True


@dataclass
class LARSConfig(BaseOptimizerConfig):
    """Layer-wise adaptive rate scaling config.

    Args:
        name: 'str', name of the optimizer.
        momentum: `float` hyperparameter >= 0 that accelerates gradient descent in
        the relevant direction and dampens oscillations. Defaults to 0.9.
        eeta: `float` LARS coefficient as used in the paper. Default set to LARS
        coefficient from the paper. (eeta / weight_decay) determines the highest
        scaling factor in LARS..
        weight_decay_rate: `float` for weight decay.
        nesterov: 'boolean' for whether to use nesterov momentum.
        classic_momentum: `boolean` for whether to use classic (or popular)
        momentum. The learning rate is applied during momentum update in classic
        momentum, but after momentum for popular momentum.
        exclude_from_weight_decay: A list of `string` for variable screening, if any
        of the string appears in a variable's name, the variable will be excluded
        for computing weight decay. For example, one could specify the list like
        ['batch_normalization', 'bias'] to exclude BN and bias from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but for
        layer adaptation. If it is None, it will be defaulted the same as
        exclude_from_weight_decay.
    """

    name: str = "LARS"
    momentum: float = 0.9
    eeta: float = 0.001
    weight_decay_rate: float = 0.0
    nesterov: bool = False
    classic_momentum: bool = True
    exclude_from_weight_decay: Optional[List[str]] = None
    exclude_from_layer_adaptation: Optional[List[str]] = None


@dataclass
class AdafactorConfig(BaseOptimizerConfig):
    """Configuration for Adafactor optimizer.

    The attributes for this class matches the arguments of the Adafactor
    implementation.
    """

    name: str = "Adafactor"
    factored: bool = True
    multiply_by_parameter_scale: bool = True
    beta1: Optional[float] = None
    decay_rate: float = 0.8
    step_offset: int = 0
    clipping_threshold: float = 1.0
    min_dim_size_to_factor: int = 128
    epsilon1: float = 1e-30
    epsilon2: float = 1e-3
    weight_decay: Optional[float] = None
    include_in_weight_decay: Optional[str] = None
