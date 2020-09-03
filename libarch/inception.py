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
"""Incpetion architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


class SmallInception(snt.Module):
  """Simple inception like network for small images (cifar/mnist)."""

  def __init__(self, num_classes=10, with_residual=False,
               large_inputs=False, name=None):
    super(SmallInception, self).__init__(name=name)
    self._num_classes = num_classes

    self._large_inputs = large_inputs
    if large_inputs:
      self.conv1 = ConvBNReLU(96, (7, 7), stride=2, name="conv1")
    else:
      self.conv1 = ConvBNReLU(96, (3, 3), name="conv1")
    self.stage1 = SmallInceptionStage([(32, 32), (32, 48)], 160,
                                      with_residual=with_residual)
    self.stage2 = SmallInceptionStage([(112, 48), (96, 64), (80, 80),
                                       (48, 96)], 240,
                                      with_residual=with_residual)
    self.stage3 = SmallInceptionStage([(176, 160), (176, 160)], 0,
                                      with_residual=with_residual)
    self._pred = snt.Linear(output_size=num_classes, name="pred")

  @tf.function
  def compute_repr(self, inputs, is_training):
    x = inputs
    if self._large_inputs:
      x = self.conv1(x, is_training)
      x = tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")
    else:
      x = self.conv1(x, is_training)
    x = self.stage1(x, is_training)
    x = self.stage2(x, is_training)
    x = self.stage3(x, is_training)

    # global pooling
    x = tf.reduce_max(x, axis=[1, 2])
    return x

  @tf.function
  def __call__(self, inputs, is_training):
    logits = self._pred(self.compute_repr(inputs, is_training))
    return logits


class MultiBranchBlock(snt.Module):
  """Simple inception-style multi-branch block."""

  def __init__(self, channels_1x1, channels_3x3, name=None):
    super(MultiBranchBlock, self).__init__(name=name)

    self.conv1x1 = ConvBNReLU(channels_1x1, (1, 1), name="conv1x1")
    self.conv3x3 = ConvBNReLU(channels_3x3, (3, 3), name="conv3x3")

  def __call__(self, inputs, is_training):
    return tf.concat([self.conv1x1(inputs, is_training),
                      self.conv3x3(inputs, is_training)], axis=3)


class SmallInceptionStage(snt.Module):
  """Stage for SmallInception model."""

  def __init__(self, mb_channels, downsample_channels, with_residual=False,
               name=None):
    super(SmallInceptionStage, self).__init__(name=name)
    self._mb_channels = mb_channels
    self._downsample_channels = downsample_channels
    self._with_residual = with_residual

    self._body = []
    for i, (ch1x1, ch3x3) in enumerate(mb_channels):
      self._body.append(MultiBranchBlock(ch1x1, ch3x3, name=f"block{i+1}"))
    if downsample_channels > 0:
      self._downsample = ConvBNReLU(downsample_channels, kernel_shape=(3, 3),
                                    stride=(2, 2), name="downsample")
    else:
      self._downsample = None

  def __call__(self, inputs, is_training):
    x = inputs
    for block in self._body:
      x = block(x, is_training)
    if self._with_residual:
      x += inputs

    if self._downsample:
      x = self._downsample(x, is_training)
    return x


class ConvBNReLU(snt.Module):
  """Conv -> BatchNorm -> ReLU."""

  def __init__(self, output_channels, kernel_shape, stride=1, rate=1,
               padding="SAME", w_init=None, name=None):
    super(ConvBNReLU, self).__init__(name=name)
    self.conv = snt.Conv2D(output_channels, kernel_shape, stride=stride,
                           rate=rate, padding=padding, with_bias=False,
                           w_init=w_init, name="conv")
    self.bn = snt.BatchNorm(True, True, name="bn")

  def __call__(self, inputs, is_training):
    x = self.conv(inputs)
    x = self.bn(x, is_training)
    return tf.nn.relu(x)
