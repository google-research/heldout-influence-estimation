"""Simple Influence / Memorization estimation on MNIST."""
import os
import itertools

import numpy as np
import numpy.random as npr

from tqdm import tqdm
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax.config import config
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Flatten, Dense, Relu, LogSoftmax

import tensorflow_datasets as tfds


def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))


def batch_correctness(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return predicted_class == target_class


def load_mnist():
  # NOTE: even when we specify shuffle_files=False, the examples 
  # in the MNIST dataset loaded by tfds is still NOT in the 
  # original MNIST data order
  raw_data = tfds.load(name='mnist', batch_size=-1,
                       as_dataset_kwargs={'shuffle_files': False})
  raw_data = tfds.as_numpy(raw_data)
  train_byte_images = raw_data['train']['image']
  train_images = train_byte_images.astype(np.float32) / 255
  train_int_labels = raw_data['train']['label']
  train_labels = one_hot(train_int_labels, 10)
  test_images = raw_data['test']['image'].astype(np.float32) / 255
  test_labels = one_hot(raw_data['test']['label'], 10)
  return dict(train_images=train_images, train_labels=train_labels,
              train_byte_images=train_byte_images, 
              train_int_labels=train_int_labels,
              test_images=test_images, test_labels=test_labels,
              test_byte_images=raw_data['test']['image'],
              test_int_labels=raw_data['test']['label'])


init_random_params, predict = stax.serial(
    Flatten,
    Dense(512), Relu,
    Dense(256), Relu,
    Dense(10), LogSoftmax)
mnist_data = load_mnist()


def subset_train(seed, subset_ratio):
  jrng = random.PRNGKey(seed)
  
  step_size = 0.1
  num_epochs = 10
  batch_size = 128
  momentum_mass = 0.9

  num_train_total = mnist_data['train_images'].shape[0]
  num_train = int(num_train_total * subset_ratio)
  num_batches = int(np.ceil(num_train / batch_size))

  rng = npr.RandomState(seed)
  subset_idx = rng.choice(num_train_total, size=num_train, replace=False)
  train_images = mnist_data['train_images'][subset_idx]
  train_labels = mnist_data['train_labels'][subset_idx]

  def data_stream(shuffle=True):
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  _, init_params = init_random_params(jrng, (-1, 28 * 28))
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  for epoch in range(num_epochs):
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))

  params = get_params(opt_state)
  trainset_correctness = batch_correctness(
      params, (mnist_data['train_images'], mnist_data['train_labels']))
  testset_correctness = batch_correctness(
      params, (mnist_data['test_images'], mnist_data['test_labels']))

  trainset_mask = np.zeros(num_train_total, dtype=np.bool)
  trainset_mask[subset_idx] = True
  return trainset_mask, np.asarray(trainset_correctness), np.asarray(testset_correctness)


def estimate_infl_mem():
  n_runs = 2000
  subset_ratio = 0.7
  
  results = []
  for i_run in tqdm(range(n_runs), desc=f'SS Ratio={subset_ratio:.2f}'):
    results.append(subset_train(i_run, subset_ratio))

  trainset_mask = np.vstack([ret[0] for ret in results])
  inv_mask = np.logical_not(trainset_mask)
  trainset_correctness = np.vstack([ret[1] for ret in results])
  testset_correctness = np.vstack([ret[2] for ret in results])

  print(f'Avg test acc = {np.mean(testset_correctness):.4f}')

  def _masked_avg(x, mask, axis=0, esp=1e-10):
    return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

  def _masked_dot(x, mask, esp=1e-10):
    x = x.T.astype(np.float32)
    return (np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), esp)).astype(np.float32)

  mem_est = _masked_avg(trainset_correctness, trainset_mask) - _masked_avg(trainset_correctness, inv_mask)
  infl_est = _masked_dot(testset_correctness, trainset_mask) - _masked_dot(testset_correctness, inv_mask)

  return dict(memorization=mem_est, influence=infl_est)
  

def show_examples(estimates, n_show=10):
  def show_image(ax, image, vmin=None, vmax=None, title=None):
    if image.ndim == 3 and image.shape[2] == 1:
      image = image.reshape((image.shape[0], image.shape[1]))
    ax.axis('off')
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    if title is not None:
      ax.set_title(title, fontsize='x-small')

  n_show = 10
  n_context1 = 4
  n_context2 = 5

  fig, axs = plt.subplots(nrows=n_show, ncols=n_context1+n_context2+1,
                          figsize=(n_context1+n_context2+1, n_show))
  idx_sorted = np.argsort(np.max(estimates['influence'], axis=1))[::-1]
  for i in range(n_show):
    # show test example
    idx_tt = idx_sorted[i]
    label_tt = mnist_data['test_int_labels'][idx_tt]
    show_image(axs[i, 0], mnist_data['test_byte_images'][idx_tt], 
               title=f'test,L={label_tt}')

    def _show_contexts(idx_list, ax_offset):
      for j, idx_tr in enumerate(idx_list):
        label_tr = mnist_data['train_int_labels'][idx_tr]
        infl = estimates['influence'][idx_tt, idx_tr]
        show_image(axs[i, j+ax_offset], mnist_data['train_byte_images'][idx_tr],
                   title=f'tr,L={label_tr},infl={infl:.3f}')

    # show training examples with highest influence
    idx_sorted_tr = np.argsort(estimates['influence'][idx_tt])[::-1]
    _show_contexts(idx_sorted_tr[:n_context1], 1)

    # show random training examples from the same class
    idx_class = np.nonzero(mnist_data['train_int_labels'] == label_tt)[0]
    idx_random = np.random.choice(idx_class, size=n_context2, replace=False)
    _show_contexts(idx_random, n_context1 + 1)

  plt.tight_layout()
  plt.savefig('mnist-examples.pdf', bbox_inches='tight')
  
  
if __name__ == '__main__':
  npz_fn = 'infl_mem.npz'
  if os.path.exists(npz_fn):
    estimates = np.load(npz_fn)
  else:
    estimates = estimate_infl_mem()
    np.savez(npz_fn, **estimates)

  show_examples(estimates)
