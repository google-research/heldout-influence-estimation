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
"""Dataset based on numpy arrays."""

import copy
import io

from . import base

from absl import logging

import numpy as np
import tensorflow as tf


class ImagesNumpy(base.Dataset):  # pylint:disable=g-classes-have-attributes
  """Images dataset loaded into memory as numpy array.

  The full data array in numpy format can be easily accessed. Suitable for
  smaller scale image datasets like MNIST (and variants), CIFAR-10 / CIFAR-100,
  SVHN, etc.

  Args:
    name: name of the dataset.
    npz_path: path to saved numpy arrays. The array names follow the
      convention of `<split>__<key>` (note: separator is double underscope).
      For example `train__image`. If a array name starts with double underscope,
      it is a meta info entry. Required meta info entries are:
      - `__num_classes`: number of classes
      - `__data_scale`: data will be divided by this scale for normalization.
        For example, this is typicall 255 for byte valued image pixels.
    random_crop: whether to perform random crop in data augmentation.
    random_fliplr: whether to perform random left-right flip in data aug.
  """

  def __init__(self, name, npz_path, random_crop=True, random_fliplr=True):
    del name  # unused

    self._random_crop = random_crop
    self._random_fliplr = random_fliplr

    if npz_path is None:
      pass  # should be filled by sub-classes
    else:
      with open(npz_path, 'rb') as in_f:
        io_buffer = io.BytesIO(in_f.read())
        arrays = np.load(io_buffer, allow_pickle=True)
        # call np.arrays to get real arrays so that they are usable
        # after the file is closed
        arrays = {k: np.array(v) for k, v in arrays.items()}

        self.ds_np = {}
        self.info = {}
        logging.info('Loading numpy dataset from %s...', npz_path)
        for key, array in arrays.items():
          if key.startswith('__'):
            # meta info
            self.info[key[2:]] = array
          else:
            split, name = key.split('__')
            if split not in self.ds_np:
              self.ds_np[split] = dict()
            self.ds_np[split][name] = array
        for split, ds in self.ds_np.items():
          logging.info('- split %s', split)
          for key, val in ds.items():
            logging.info('  * %s: %s', key, str(val.shape))
        logging.info('Dataset loaded.')

    self._add_index_feature()

  def _add_index_feature(self):
    """add 'index' feature if not present."""
    for split in self.ds_np:
      if 'index' in self.ds_np[split]:
        continue
      n_sample = len(self.ds_np[split]['label'])
      index = np.arange(n_sample)
      self.ds_np[split]['index'] = index

  @property
  def num_classes(self):
    return int(self.info['num_classes'])

  @property
  def data_scale(self):
    return self.info['data_scale']

  def get_num_examples(self, split_name):
    return self.ds_np[split_name]['image'].shape[0]

  def normalize_images(self, batch_image_np):
    return batch_image_np.astype(np.float32) / self.data_scale

  @staticmethod
  def random_crop(batch_image_np, pad=4):
    """Randomly cropping images for data augmentation."""
    n, h, w, c = batch_image_np.shape
    # pad
    padded_image = np.zeros((n, h+2*pad, w+2*pad, c),
                            dtype=batch_image_np.dtype)
    padded_image[:, pad:-pad, pad:-pad, :] = batch_image_np
    # crop
    idxs = np.random.randint(2*pad, size=(n, 2))
    cropped_image = np.array([
        padded_image[i, y:y+h, x:x+w, :]
        for i, (y, x) in enumerate(idxs)])
    return cropped_image

  @staticmethod
  def random_fliplr(batch_image_np):
    """Randomly do left-right flip on images."""
    n = batch_image_np.shape[0]
    coins = np.random.choice([-1, 1], size=n)
    flipped_image = np.array([
        batch_image_np[i, :, ::coins[i], :]
        for i in range(n)])
    return flipped_image

  def iterate(self, split_name, batch_size, shuffle=False, augmentation=False,
              subset_index=None):
    n_sample = self.get_num_examples(split_name)
    # make a shallow copy
    dset = copy.copy(self.ds_np[split_name])

    if subset_index is not None:
      n_sample = len(subset_index)
      for key in dset.keys():
        dset[key] = dset[key][subset_index]

    if shuffle:
      rp = np.random.permutation(n_sample)
      for key in dset.keys():
        dset[key] = dset[key][rp]

    for i in range(0, n_sample, batch_size):
      batch = {key: val[i:i+batch_size]
               for key, val in dset.items()}
      batch['image'] = self.normalize_images(batch['image'])
      if augmentation:
        if self._random_crop:
          batch['image'] = self.random_crop(batch['image'])
        if self._random_fliplr:
          batch['image'] = self.random_fliplr(batch['image'])

      batch = {key: tf.convert_to_tensor(val) for key, val in batch.items()}
      yield batch
