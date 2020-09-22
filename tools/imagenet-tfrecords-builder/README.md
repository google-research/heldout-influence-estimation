# Building ImageNet Dataset with Index Information

In order to talk about the influence and memorization of each 
example, it is more convenient to have an index / id for each example.
The standard ImageNet dadaset does not come with a natural order for the 
examples. We have chosen to use an arbitrary ordering to generate 
an index for each example. However, due to the copyright of ImageNet data,
we cannot release the dataset with index for public download.

Instead, we release a mapping from image filenames to their index for
the user to reconstruct the data with index themselves. This doc describe
the tool to build a tfrecords dataset with index that can be directly
used in our open sourced data pipeline.

This script is adapted from [kmonachopoulos/ImageNet-to-TFrecord](https://github.com/kmonachopoulos/ImageNet-to-TFrecord).


## Preparation

The following files can be found in this folder:

- build_dataset.py
- imagenet_lsvrc_2015_synsets.txt
- imagenet_metadata.txt

You will need the [ImageNet filename to index mapping](https://pluskid.github.io/influence-memorization/data/imagenet_index.npz)
information that can be downloaded from [this page](https://pluskid.github.io/influence-memorization/).

You will also need the [original ImageNet ILSVRC 2012 data
files](http://www.image-net.org/). See the comments at the beginning of
`build_dataset.py` for how the ImageNet image files should be organized.
Basically the folder structur of the original ImageNet dataset should work.

## Dataset Building

This command build the tf records files for the training set:

```
python build_dataset.py --train_directory=/path/to/imagenet/train --output_directory=imagenet-tfrecords/
```

Specify `--validation_directory` to build the validation set. The validation set is much smaller,
so it is good to try to build it so see if everything works properly. See `build_dataset.py` for
customization of other options.

Note by default the validation set will be sharded into 128 files and the training set 1024 files.

## Using the Dataset

After building the tfrecord files with index information, you can place them in a suitable place.
Go to the root of this repository, find `libdata/indexed_tfrecords.py`, and find the global variable
`PREDEFINED_DATA`, modify the `filenames` property for ImageNet to point to the files you just 
generated.

After that, you should be able to use this dataset via the `indexed_tfrecords.IndexedImageDataset`
data provider. Please see `imagenet_demo()` in `demo.py` at the root of this repository for more details.
