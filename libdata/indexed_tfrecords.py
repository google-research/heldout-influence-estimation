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
"""Dataset based on loading from tfrecords files with `index` feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from . import base
from . import imagenet_preprocessing

from absl import logging

import tensorflow.compat.v2 as tf


_SHUFFLE_BUFFER = 10000
DEFAULT_IMAGE_SIZE = 224
PREDEFINED_DATA = {
    'tiny_imagenet': {
        'num_classes': 200,
        'num_examples': {'train': 200*500, 'test': 200*50},
        'filenames': {
            'train': ['/path/to/tiny_imagenet/train.tfrecords'],
            'test': ['/path/to/tiny_imagenet/val.tfrecords']
        },
        'feature_names': {
            'image_raw': 'image_raw',
            'label': 'label',
            'index': 'index',
        },
    },
    'imagenet': {
        'num_classes': 1000,
        'num_examples': {'train': 1281167, 'test': 50000},
        'filenames': {
            'train': [f'/path/to/imagenet-indexed/train-{i:05d}-of-01024'
                      for i in range(1024)],
            'test': [f'/path/to/imagenet-indexed/validation-{i:05d}-of-00128'
                     for i in range(128)]
        },
        'feature_names': {
            'image_raw': 'image/encoded',
            'label': 'image/class/label',
            'index': 'index',
            'filename': 'image/filename',
        },
        'normalizer': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
    },
}


class IndexedImageDataset(base.Dataset):
  """Dataset that directly load from tfrecords files."""

  def __init__(self, name, meta=None, include_image_buffer=False,
               include_filename=False):
    """Construct IndexedImageDataset.

    Args:
      name: name of the dataset. Will also be used as key to fetch
        pre-defined meta data for the dataset if exists.
      meta: if provided, will override the pre-fetched meta. If the
        name does not correspond to any pre-defined dataset, then
        the meta must be explicitly provided.
      include_image_buffer: if True, the returned data dict will
        include an entry `image_buffer` containing the raw JPEG
        bytes.
      include_filename: if True, the returned data dict will include
        an entry `filename` containing the filename of each image.

    Raises:
      KeyError: if meta cannot be determined from either `name` or
        `meta`.
    """
    dset_meta = PREDEFINED_DATA.get(name, None) or meta
    if dset_meta is None:
      raise KeyError(
          f'Unknown tfrecord dataset {name}! '
          'If not one of {",".join(PREDEFINED_DATA.keys()}}, '
          'then meta should be provided explicitly.')
    self.meta = dset_meta
    self.include_image_buffer = include_image_buffer
    self.include_filename = include_filename
    self.cached_dataset = dict()
    self._use_onehot_label = False

  @property
  def num_classes(self):
    return self.meta['num_classes']

  @property
  def use_onehot_label(self):
    return self._use_onehot_label

  def get_num_examples(self, split_name):
    return self.meta['num_examples'][split_name]

  def get_tf_dataset(self, split_name, batch_size,
                     shuffle=False, augmentation=False,
                     subset_index=None):
    cache_key = (split_name, batch_size, shuffle, augmentation)
    # perform caching only when subset_index is None
    if subset_index is None and cache_key in self.cached_dataset:
      dset_obj = self.cached_dataset[cache_key]
    else:
      filenames = self.meta['filenames'][split_name]
      dset_obj = load_data(
          filenames, batch_size,
          shuffle=shuffle, augmentation=augmentation,
          include_image_buffer=self.include_image_buffer,
          include_filename=self.include_filename,
          subset_index=subset_index,
          feature_names=self.meta['feature_names'],
          mean_std=self.meta.get('normalizer', None))
      if subset_index is None:
        self.cached_dataset[cache_key] = dset_obj

    return dset_obj

  def iterate(self, split_name, batch_size,
              shuffle=False, augmentation=False,
              subset_index=None):
    yield from self.get_tf_dataset(
        split_name, batch_size, shuffle, augmentation, subset_index)


def process_record_dataset(dataset,
                           shuffle,
                           augmentation,
                           batch_size,
                           shuffle_buffer,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           drop_remainder=False,
                           tf_data_experimental_slack=False,
                           post_processor=None,
                           subset_index=None,
                           include_image_buffer=False,
                           include_filename=False,
                           feature_names=None,
                           mean_std=None):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    shuffle: A boolean denoting whether to shuffle the data.
    augmentation: A boolean denoting whether to perform data augmentation.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    dtype: Data type to use for images/features.
    datasets_num_private_threads: Number of threads for a private
      threadpool created for all datasets computation.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's
      `experimental_slack` option.
    post_processor: if not None, a function that do post processing on examples.
    subset_index: if not None, contain a list of indexes to filter the data,
      so that only data samples whose index listed here will be used.
    include_image_buffer: if True, include `image_buffer` in the feature
      dict containing the raw JPEG bytes.
    include_filename: if True, include `filename` in the feature dict containing
      the filename of each example.
    feature_names: if not None, a dict containing the feature keys for
      `image_raw`, `label` and `index` to be extracted from the tfrecords file.
    mean_std: if not None, a dict with 'mean' and 'std' used to normalize the
      input pixels (which are already in the range of [0, 1]).

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  # Defines a specific size thread pool for tf.data operations.
  if datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = (
        datasets_num_private_threads)
    dataset = dataset.with_options(options)
    logging.info(
        'datasets_num_private_threads: %s', datasets_num_private_threads)

  # Disable intra-op parallelism to optimize for throughput instead of latency.
  options = tf.data.Options()
  options.experimental_threading.max_intra_op_parallelism = 1
  dataset = dataset.with_options(options)

  # Prefetches a batch at a time to smooth out the time taken to load input
  # files for shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if shuffle:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # Parses the raw records into images and labels.
  def do_parse_record(value):
    return parse_record(
        value, augmentation, dtype, post_processor,
        include_image_buffer=include_image_buffer,
        include_filename=include_filename,
        feature_names=feature_names, mean_std=mean_std)
  dataset = dataset.map(
      do_parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if subset_index is not None:
    subset_index_tensor = tf.constant(subset_index)
    def predicate(example):
      return tf.reduce_any(tf.equal(example['index'], subset_index_tensor))
    dataset = dataset.filter(predicate)

  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  if tf_data_experimental_slack:
    options = tf.data.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)

  return dataset


def parse_record(raw_record, augmentation, dtype, post_processor=None,
                 include_image_buffer=False, include_filename=False,
                 feature_names=None, mean_std=None):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    augmentation: A boolean denoting whether to perform data augmentation.
    dtype: data type to use for images/features.
    post_processor: if not None, a function that takes the example dict,
      and return the post processed dict.
    include_image_buffer: if True, include `image_buffer` in the feature
      dict containing the raw JPEG bytes.
    include_filename: if True, include `filename` in the feature dict
      containing the filename of each example.
    feature_names: if not None, a dict containing the feature keys for
      `image_raw`, `label` and `index` to be extracted from the tfrecords file.
    mean_std: if not None, a dict with 'mean' and 'std' used to normalize the
      input pixels (which are already in the range of [0, 1]).

  Returns:
    Dict with processed image tensor (image), label tensor (label), and
    index (index).
  """
  image_buffer, label, index, filename = _parse_example_proto(
      raw_record, feature_names, include_filename=include_filename)

  image = imagenet_preprocessing.preprocess_image(
      image_bytes=image_buffer,
      is_training=augmentation,
      image_size=DEFAULT_IMAGE_SIZE,
      use_bfloat16=dtype == tf.bfloat16,
      mean_std=mean_std)

  # Subtract one so that labels are in [0, 1000), and cast to int64
  label = tf.cast(label - 1, dtype=tf.int64)
  result = dict(image=image, label=label, index=index)
  if include_image_buffer:
    result['image_buffer'] = image_buffer
  if include_filename:
    result['filename'] = filename
  if post_processor is not None:
    result = post_processor(result)
  return result


def _parse_example_proto(example_serialized, feature_names,
                         include_filename=False):
  """Parses an Example proto containing a training example of an image.

  Each Example proto contains the following fields (values are
  included as examples), the keys could be re-mapped by `feature_names`:

    image_raw: <JPEG encoded string>
    label: 615 (1 ~ num_classes)
    index: 713 (0 ~ num_examples-1)

  NOTE: to be consistent with pre-existing ImageNet tfrecords, the labels
  are indexed starting from 1 instead of 0. In the processing above, we
  explicitly subtract 1 from the loaded labels to create 0~(num_class-1)
  numerical range.

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    feature_names: if not None, a dict containing the feature keys for
      `image_raw`, `label` and `index` to be extracted from the tfrecords file.
    include_filename: if True, also parse the filename, otherwise, the
      filename return value will be None.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    index: tf.int64 containing the index.
    filename: Tensor tf.string if `include_filename` is True, otherwise None.
  """
  # Dense features in Example proto.
  if feature_names is None:
    feature_names = {}
  image_name = feature_names.get('image_raw', 'image_raw')
  label_name = feature_names.get('label', 'label')
  index_name = feature_names.get('index', 'index')

  feature_map = {
      image_name: tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
      label_name: tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      index_name: tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
  }
  if include_filename:
    filename_name = feature_names.get('filename', 'filename')
    feature_map[filename_name] = tf.io.FixedLenFeature(
        [], dtype=tf.string, default_value='')

  features = tf.io.parse_single_example(serialized=example_serialized,
                                        features=feature_map)
  label = tf.cast(features[label_name], dtype=tf.int32)

  if include_filename:
    filename = features[filename_name]
  else:
    filename = None

  return features[image_name], label, features[index_name], filename


def load_data(filenames, batch_size, shuffle=False, augmentation=False,
              dtype=tf.float32,
              datasets_num_private_threads=None,
              drop_remainder=False,
              tf_data_experimental_slack=False,
              post_processor=None,
              subset_index=None,
              include_image_buffer=False,
              include_filename=False,
              feature_names=None,
              mean_std=None):
  """Input function which provides batches for train or eval.

  Args:
    filenames: list of tfrecords files.
    batch_size: The number of samples per batch.
    shuffle: A boolean denoting whether to shuffle the data.
    augmentation: A boolean denoting whether to perform data augmentation.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's
      `experimental_slack` option.
    post_processor: if not None, a function that do post processing on examples.
    subset_index: if not None, contain a list of indexes to filter the data,
      so that only data samples whose index listed here will be used.
    include_image_buffer: if True, include `image_buffer` in the feature
      dict containing the raw JPEG bytes.
    include_filename: if True, include `filename` in the feature dict containing
      the filename of the example.
    feature_names: if not None, a dict containing the feature keys for
      `image_raw`, `label` and `index` to be extracted from the tfrecords file.
    mean_std: if not None, a dict with 'mean' and 'std' used to normalize the
      input pixels (which are already in the range of [0, 1]).

  Returns:
    A dataset that can be used for iteration.
  """
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if shuffle:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=len(filenames))

  # Convert to individual records.
  # cycle_length = 10 means that up to 10 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=10,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return process_record_dataset(
      dataset=dataset,
      shuffle=shuffle,
      augmentation=augmentation,
      batch_size=batch_size,
      shuffle_buffer=_SHUFFLE_BUFFER,
      dtype=dtype,
      datasets_num_private_threads=datasets_num_private_threads,
      drop_remainder=drop_remainder,
      tf_data_experimental_slack=tf_data_experimental_slack,
      post_processor=post_processor,
      subset_index=subset_index,
      include_image_buffer=include_image_buffer,
      include_filename=include_filename,
      feature_names=feature_names,
      mean_std=mean_std)

