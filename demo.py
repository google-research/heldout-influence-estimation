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
"""A demo script showing how to load and investigate checkpoints."""
import os

import libarch
import libdata
import libutil

import numpy as np
import tensorflow as tf


def load_data(name, provider='tfds.TFDSImagesNumpy', kwargs=None):
  """Load dataset."""
  kwargs = kwargs or dict()
  kwargs['name'] = name
  ctor = libutil.rgetattr(libdata, provider)
  return ctor(**kwargs)


def build_model(num_classes, arch, kwargs=None):
  """Build model according to architecture name."""
  kwargs = kwargs or dict()
  kwargs['num_classes'] = num_classes
  ctor = libutil.rgetattr(libarch, arch)
  return ctor(**kwargs)


def load_checkpoint(model, checkpoint_dir):
  """Load the latest checkpoint."""
  v_epoch = tf.Variable(0, dtype=tf.int64, name='epoch', trainable=False)
  v_gs = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
  checkpoint = tf.train.Checkpoint(model=model, epoch=v_epoch, global_step=v_gs)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint=checkpoint, directory=checkpoint_dir, max_to_keep=5)
  if checkpoint_manager.latest_checkpoint:
    # use expect_partial to silence warnings related optimizer variables
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
  else:
    raise KeyError(f'No checkpoint found at {checkpoint_dir}')
  return dict(epoch=int(v_epoch.numpy()), global_step=int(v_gs.numpy()),
              path=checkpoint_manager.latest_checkpoint)


def do_eval(model, dataset, split='test', batch_size=200):
  """Evaluation a model on a given dataset."""
  correctness_all = []
  index_all = []
  for inputs in dataset.iterate(split, batch_size,
                                shuffle=False, augmentation=False):
    predictions = model(inputs['image'], is_training=False)
    correctness_all.append(tf.equal(tf.argmax(predictions, axis=1),
                                    inputs['label']).numpy())
    index_all.append(inputs['index'].numpy())

  correctness_all = np.concatenate(correctness_all, axis=0)
  index_all = np.concatenate(index_all, axis=0)

  return dict(correctness=correctness_all, index=index_all)


def cifar10_demo(model_dir, arch='inception.SmallInception', split='test'):
  """Demo for CIFAR-10."""
  dataset = load_data('cifar10:3.0.2')
  model = build_model(dataset.num_classes, arch)
  load_results = load_checkpoint(model, os.path.join(model_dir, 'checkpoints'))

  aux_arrays = np.load(os.path.join(model_dir, 'aux_arrays.npz'))
  subsample_tr_idx = aux_arrays['subsample_idx']
  print(f'Loaded from checkpoint (epoch={load_results["epoch"]}, ' +
        f'global_step={load_results["global_step"]}) trained from a random ' +
        f'{len(subsample_tr_idx)/dataset.get_num_examples("train")*100:.0f}%' +
        ' subset of training examples.')

  results = do_eval(model, dataset, split=split)
  print(f'Eval accuracy on {split} = {np.mean(results["correctness"]):.4f}')

  def ordered_correctness(correctness, index):
    new_correctness = np.zeros_like(correctness)
    new_correctness[index] = correctness
    return new_correctness

  # make sure the evaluation correctness matches the exported result
  oc1 = ordered_correctness(results['correctness'], results['index'])
  oc2 = ordered_correctness(aux_arrays[f'correctness_{split}'],
                            aux_arrays[f'index_{split}'])
  assert np.all(oc1 == oc2)


if __name__ == '__main__':
  cifar10_demo('/path/to/individual/run')
