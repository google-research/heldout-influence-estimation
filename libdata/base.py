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
"""Base class for Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Dataset(abc.ABC):
  """The dataset base class."""

  @property
  @abc.abstractmethod
  def num_classes(self):
    pass

  @abc.abstractmethod
  def get_num_examples(self, split_name):
    pass

  @abc.abstractmethod
  def iterate(self, split_name, batch_size, shuffle=False, augmentation=False,
              subset_index=None):
    """Iterate over the data. See get_tf_dataset for args docs."""

  def get_tf_dataset(self, split_name, batch_size,
                     shuffle=False, augmentation=False,
                     subset_index=None):
    """Get a tf.data.Dataset instance, if available.

    This method is not marked as abstract as the subclass could choose to
    not implement this.

    Args:
      split_name: name of the data split to get.
      batch_size: batch size.
      shuffle: whether to shuffle the data.
      augmentation: whether to perform data augmentation.
      subset_index: if not None, the user could be providing a numpy array
        containing the index for the examples to be included. This is useful
        when training on a subset of examples. This feature is not frequently
        used. If a subclass decides to not implement this, an exception should
        be raised instead of ignoring it silently.

    Returns:
      A tf.data.Dataset instance.

    Raises:
      NotImplementedError if the dataset is internally not implemented
      with the tf.data pipeline.
    """
    raise NotImplementedError()
