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
"""TFDS based dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import numpy as np_dset

import tensorflow_datasets as tfds


class TFDSImagesNumpy(np_dset.ImagesNumpy):
  """TFDS Images dataset loaded into memory as numpy array.

  The full data array in numpy format can be easily accessed. Suitable for
  smaller scale image datasets like MNIST (and variants), CIFAR-10 / CIFAR-100,
  SVHN, etc.
  """

  def __init__(self, name, **kwargs):
    self.ds, self.info = tfds.load(name, batch_size=-1,
                                   as_dataset_kwargs={'shuffle_files': False},
                                   with_info=True)
    self.ds_np = tfds.as_numpy(self.ds)

    kwargs['npz_path'] = None
    super(TFDSImagesNumpy, self).__init__(name, **kwargs)

  @property
  def data_scale(self):
    return 255.0

  @property
  def num_classes(self):
    return self.info.features['label'].num_classes
