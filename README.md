# Heldout Training Checkpoints

This repository contains code to load and inspect the models trained with a random heldout
subset of the training set. The predictions of those models can be used to estimate useful
statistics such as *memorization*, *influence*, and *consistency scores*, as demonstrated
in the following papers.

* Vitaly Feldman♮, Chiyuan Zhang♮.
  [What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation](https://arxiv.org/abs/2008.03703). ♮equal contribution. 2020.
  【[project website](https://pluskid.github.io/influence-memorization/)】
* Ziheng Jiang♮, Chiyuan Zhang♮, Kunal Talwar, Michael C. Mozer.
  [Characterizing Structural Regularities of Labeled Data in Overparameterized Models](https://arxiv.org/abs/2002.03206). ♮equal contribution. 2020.
  【[project website](https://pluskid.github.io/structural-regularity/)】
                                                                                                                                                                                                              In addition to the pre-computed statistics released in the project website of each respective paper,
we also release the model checkpoints from those pre-trained models to facilitate future research.
## Dependencies

This code depends on the following open source libraries

* Tensorflow 2
* [Tensorflow Datasets](https://www.tensorflow.org/datasets)
* [Sonnet](https://github.com/deepmind/sonnet)

## Model Checkpoints

The model checkpoints can be downloaded from an external source. Each *experiment* consists of
a collection of checkpoints for a certain neural network architecture trained (with the
heldout procedure) on a certain dataset. The files are organized in the following way:

```
<experiment_name (e.g. cifar10-inception)>
  +-- <subset_ratio (e.g. 0.1)
  |     +-- <run_id (e.g. 0)>
  |     |     +-- aux_arrays.npz
  |     |     +-- checkpoints
  |     |           +-- <ckpt-epoch>
  |     +-- 1                                                                                                                                                                                                   |     +-- ...
  +-- 0.2
  +-- ...
```

Depending on the experiment, there might be multiple subset ratios, and a different number
of runs for each subset ratio. The `<ckpt-epoch>` is the checkpoint files in the Tensorflow
format for the final training epoch. The `aux_arrays.npz` is a numpy array exported file
containing the following information of each experiment:

- `correctness_<split>`: a binary indicator array of the prediction correctness of the final
  trained model, where the `split` could be `train`, `removed` and `test`. Here `train` + `removed`
  is the full training set of the original dataset.
- `index_<split>`: an integer array for the index of each example in the `correctness_<split>` array.
  This is useful when counting statistics across different experiments where the evaluation might
  be done in different orders. Note for `train` and `removed`, the index indicates the id of each
  example in the original full training set, therefore compatible across different experiments.
- `subsample_idx`: the list of indices of examples from the original training set that is
  selected to be used in this run.

## Code Demo

In the file `demo.py`, we include some demo code showing how to construct models, load checkpoints,
run the models for evaluation and cross check with the results found in `aux_arrays.npz`.
