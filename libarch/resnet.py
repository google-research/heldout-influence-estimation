# Lint as: python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ResNets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Mapping, Optional, Sequence, Text, Union

import sonnet as snt
import tensorflow as tf


class ResNetV2(snt.Module):
  """ResNet model."""

  def __init__(self,
               num_classes: int,
               blocks_per_group_list: Sequence[int],
               bottleneck_block: bool,
               bn_config: Optional[Mapping[Text, float]] = None,
               small_input: bool = False,
               channels_per_group_list: Sequence[int] = (64, 128, 256, 512),
               name: Optional[Text] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      blocks_per_group_list: A sequence of length 4 that indicates the number of
        blocks created in each group.
      bottleneck_block: whether to use bottleneck block or basic block.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers. By default the `decay_rate` is
        `0.9` and `eps` is `1e-5`.
      small_input: Whether designed for small (CIFAR) inputs or large (ImageNet)
        inputs.
      channels_per_group_list: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      name: Name of the module.
    """
    super(ResNetV2, self).__init__(name=name)
    if bn_config is None:
      bn_config = {"decay_rate": 0.9, "eps": 1e-5}
    self._bn_config = bn_config
    self._small_input = small_input
    self._bottleneck_block = bottleneck_block

    # Number of blocks in each group for ResNet.
    if len(blocks_per_group_list) != len(channels_per_group_list):
      raise ValueError(
          "`blocks_per_group_list` must be of the same length "
          "as `channels_per_group_list`")
    self._blocks_per_group_list = blocks_per_group_list
    self._channels_per_group_list = channels_per_group_list

    if self._small_input:
      self._initial_conv = snt.Conv2D(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          with_bias=False,
          padding=snt.pad.same,
          name="initial_conv")
    else:
      self._initial_conv = snt.Conv2D(
          output_channels=64,
          kernel_shape=7,
          stride=2,
          with_bias=False,
          padding=snt.pad.same,
          name="initial_conv")

    self._block_groups = []
    strides = [2 for _ in self._channels_per_group_list]
    strides[0] = 1
    for i in range(len(self._channels_per_group_list)):
      self._block_groups.append(
          BlockGroup(
              channels=self._channels_per_group_list[i],
              num_blocks=self._blocks_per_group_list[i],
              stride=strides[i],
              bn_config=bn_config,
              bottleneck_block=self._bottleneck_block,
              name="block_group_%d" % i))

    self._final_batchnorm = snt.BatchNorm(
        create_scale=True,
        create_offset=True,
        name="final_batchnorm",
        **bn_config)

    self._logits = snt.Linear(
        output_size=num_classes, w_init=snt.initializers.Zeros(), name="logits")

  def compute_repr(self, inputs, is_training):
    net = inputs
    net = self._initial_conv(net)

    if not self._small_input:
      net = tf.nn.max_pool2d(
          net, ksize=3, strides=2, padding="SAME", name="initial_max_pool")

    for block_group in self._block_groups:
      net = block_group(net, is_training)

    net = self._final_batchnorm(net, is_training=is_training)
    net = tf.nn.relu(net)
    net = tf.reduce_mean(net, axis=[1, 2], name="final_avg_pool")
    return net

  def __call__(self, inputs, is_training):
    net = self.compute_repr(inputs, is_training=is_training)
    return self._logits(net)


class BlockGroup(snt.Module):
  """Higher level block for ResNet implementation."""

  def __init__(self,
               channels: int,
               num_blocks: int,
               stride: Union[int, Sequence[int]],
               bn_config: Mapping[Text, float],
               bottleneck_block: bool = False,
               name: Optional[Text] = None):
    super(BlockGroup, self).__init__(name=name)
    self._channels = channels
    self._num_blocks = num_blocks
    self._stride = stride
    self._bn_config = bn_config

    if bottleneck_block:
      block = BottleNeckBlockV2
    else:
      block = BasicBlockV2

    self._blocks = []
    for id_block in range(num_blocks):
      self._blocks.append(
          block(
              channels=channels,
              stride=stride if id_block == 0 else 1,
              use_projection=(id_block == 0),
              bn_config=bn_config,
              name="block_%d" % id_block))

  def __call__(self, inputs, is_training):
    net = inputs
    for block in self._blocks:
      net = block(net, is_training=is_training)
    return net


class BottleNeckBlockV2(snt.Module):
  """Bottleneck Block for a Resnet implementation."""

  def __init__(self,
               channels: int,
               stride: Union[int, Sequence[int]],
               use_projection: bool,
               bn_config: Mapping[Text, float],
               name: Optional[Text] = None):
    super(BottleNeckBlockV2, self).__init__(name=name)
    self._channels = channels
    self._stride = stride
    self._use_projection = use_projection
    self._bn_config = bn_config

    batchnorm_args = {"create_scale": True, "create_offset": True}
    batchnorm_args.update(bn_config)

    if self._use_projection:
      self._proj_conv = snt.Conv2D(
          output_channels=channels * 4,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding=snt.pad.same,
          name="shortcut_conv")

    self._conv_0 = snt.Conv2D(
        output_channels=channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding=snt.pad.same,
        name="conv_0")

    self._bn_0 = snt.BatchNorm(name="batchnorm_0", **batchnorm_args)

    self._conv_1 = snt.Conv2D(
        output_channels=channels,
        kernel_shape=3,
        stride=stride,
        with_bias=False,
        padding=snt.pad.same,
        name="conv_1")

    self._bn_1 = snt.BatchNorm(name="batchnorm_1", **batchnorm_args)

    self._conv_2 = snt.Conv2D(
        output_channels=channels * 4,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding=snt.pad.same,
        name="conv_2")

    # NOTE: Some implementations of ResNet50 v2 suggest initializing gamma/scale
    # here to zeros.
    self._bn_2 = snt.BatchNorm(name="batchnorm_2", **batchnorm_args)

  def __call__(self, inputs, is_training):
    net = inputs
    shortcut = inputs

    for i, (conv_i, bn_i) in enumerate(((self._conv_0, self._bn_0),
                                        (self._conv_1, self._bn_1),
                                        (self._conv_2, self._bn_2))):
      net = bn_i(net, is_training=is_training)
      net = tf.nn.relu(net)
      if i == 0 and self._use_projection:
        shortcut = self._proj_conv(net)
      net = conv_i(net)

    return net + shortcut


class BasicBlockV2(snt.Module):
  """Basic Block for a Resnet implementation."""

  def __init__(self,
               channels: int,
               stride: Union[int, Sequence[int]],
               use_projection: bool,
               bn_config: Mapping[Text, float],
               name: Optional[Text] = None):
    super(BasicBlockV2, self).__init__(name=name)
    self._channels = channels
    self._stride = stride
    self._use_projection = use_projection
    self._bn_config = bn_config

    batchnorm_args = {"create_scale": True, "create_offset": True}
    batchnorm_args.update(bn_config)

    if self._use_projection:
      self._proj_conv = snt.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding=snt.pad.same,
          name="shortcut_conv")

    self._conv_0 = snt.Conv2D(
        output_channels=channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding=snt.pad.same,
        name="conv_0")

    self._bn_0 = snt.BatchNorm(name="batchnorm_0", **batchnorm_args)

    self._conv_1 = snt.Conv2D(
        output_channels=channels,
        kernel_shape=3,
        stride=stride,
        with_bias=False,
        padding=snt.pad.same,
        name="conv_1")

    self._bn_1 = snt.BatchNorm(name="batchnorm_1", **batchnorm_args)

  def __call__(self, inputs, is_training):
    net = inputs
    shortcut = inputs

    for i, (conv_i, bn_i) in enumerate(((self._conv_0, self._bn_0),
                                        (self._conv_1, self._bn_1))):
      net = bn_i(net, is_training=is_training)
      net = tf.nn.relu(net)
      if i == 0 and self._use_projection:
        shortcut = self._proj_conv(net)
      net = conv_i(net)

    return net + shortcut


class ResNet50(ResNetV2):

  def __init__(self, num_classes, name=None):
    super(ResNet50, self).__init__(
        num_classes=num_classes,
        blocks_per_group_list=[3, 4, 6, 3],
        bottleneck_block=True,
        small_input=False,
        name=name)


class CifarResNet50(ResNetV2):
  """CifarResNet50 module."""

  def __init__(self, num_classes, name=None):
    super(CifarResNet50, self).__init__(
        num_classes=num_classes,
        blocks_per_group_list=[3, 4, 6, 3],
        bottleneck_block=True,
        small_input=True,
        name=name)


class CifarResNet18(ResNetV2):
  """CifarResNet18 module."""

  def __init__(self, num_classes, name=None):
    super(CifarResNet18, self).__init__(
        num_classes=num_classes,
        blocks_per_group_list=[2, 2, 2, 2],
        bottleneck_block=False,
        small_input=True,
        name=name)


