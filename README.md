# Radial

`Radial` is a framework for constructing neural network solutions to the problem of finding the exit point of the well to the radial mode.

Main features:

* load data from XLSX and save it to NPZ format.
* drop outliers using Isolation Forest.
* normalize, drop negatives, make a point approximation by 1d interpolation and logarithm of the time and derivative of the pressure.
* making a cross validation approach to finding an optimal parameters of the neural network.
* Train with models from zoo of state-of-the-art neural networks.
* Predict on logarithmic data of NPZ format.

## About Radial

> Radial is based on [BatchFlow](https://github.com/analysiscenter/batchflow).

Radial has three modules [``core``](), [``preprocessing``]() and [``pipelines``]().

``core`` module contains ``RadialBatch`` class. This class include actions for loading data, make an normalization, approximation and others points preprocessing. These actions allows to create workflow that could be used to training machine learning models or neural networks.

``preprocessing`` module is designed to work with raw data. This module has two files for preprocessing:
* ``xls_to_npz.py`` - comprise a function to convert raw data from XLS format to NPZ format
* ``make_isolation.py`` - filtering outliers using Isolation Forest algorithm.

``pipelines`` module provides predefined workflows to:
* preprocess time and derivative of the pressure.
* train a model and perform an inference to finding the exit point of the well to the radial mode.

## Basic usage

### Preprocessing

#### Preprocessing xls files

To prepare data that stored in XLSX just run following commands:
```bash
foo@bar:~$ python xls_to_npz.py -l path/to/whole_data.xlsx -s path/to/save
Done!
```

If you have a different files with train and test, use following command:
```bash
foo@bar:~$ python xls_to_npz.py -l path/to/TEST_data.xlsx path/to/TRAIN_data.xlsx  -s path/to/save
Done!
```

or if you want to save test and train path of data to different directories.

```bash
foo@bar:~$ python xls_to_npz.py -l path/to/TEST_data.xlsx path/to/TRAIN_data.xlsx  -s path/to/TEST_save path/to/TRAIN_save
Done!
```

#### Removing outliers

The next step is optional. If you have a large dataset than dropping outliers in the pipeline could slow train process, to avoid this problem use `drop_outliers.py`. In the same time, this function available in the ``RadialBatch``.
Anyway, following command allows you to run this function with NPZ-data:
```bash
foo@bar:~$ python drop_outliers.py -l path/to/npz_data -s path/to/save
Done!
```

As in previous time, `drop outliers.py` might be used with several paths:
```bash
foo@bar:~$ python drop_outliers.py -l path/to/npz_TEST_data path/to/npz_TEST_data -s path/to/save
Done!
```

or if you want to save test and train path of data to different directories.

```bash
foo@bar:~$ python drop_outliers.py -l path/to/npz_TEST_data path/to/npz_TRAIN_data -s path/to/TEST_save path/to/TRAIN_save
Done!
```

### Train model

Here is an example of pipeline that load data, makes preprocessing and trains a model for 100 epochs:
```python
train_pipeline = (
    Pipeline()
    .load(fmt='npz')
    .drop_negative(src=['time', 'derivative'])
    .drop_outliers(src=['time', 'derivative'])
    .to_log10(src=['time', 'derivative', 'target'],
              dst=['time', 'derivative', 'target'])
    .normalize(src=['time', 'derivative', 'target'],
               dst=['time', 'derivative', 'target'],
               src_range=[None, None, 'derivative_q'],
               dst_range=[None, 'derivative_q', None])
    .get_samples(n_samples, n_samples=1, sampler=sampler, src=['time', 'derivative'])
    .make_points(src=['time', 'derivative'], dst=['points'])
    .make_target(src='target')
    .init_variable('loss', init_on_each_run=list)
    .init_model('dynamic', RadialModel, model_name, config=model_config)
    .train_model('model', fetches='loss', feed_dict=feed_dict,
                 save_to=V('loss'), mode='w')
) << data

train_pipeline.run(50, n_epochs=100, drop_last=True, shuffle=True, bar=True)
```

## Installation

### Installation as a python package

With [pipenv](https://docs.pipenv.org/):

    pipenv install git+https://github.com/analysiscenter/radial.git#egg=radial

With [pip](https://pip.pypa.io/en/stable/):

    pip3 install git+https://github.com/analysiscenter/radial.git

After that just import `radial`:
```python
import radial
```

### Installation as a project repository

When cloning repo from GitHub use flag ``--recursive`` to make sure that ``batchflow`` submodule is also cloned.

    git clone --recursive https://github.com/analysiscenter/radial.git


## Citing Radial

Please cite Radial in your publications if it helps your research.


    Khudorozhkov R., Broilovskiy A., Mylzenova D., Podvyaznikov D. Radial library for deep research
    of finding exit point to the radial mode. 2019.

```
@misc{cardio_2017_1156085,
  author       = {Khudorozhkov R., Broilovskiy A., Mylzenova D., Podvyaznikov D.},
  title        = {Radial library for deep research of finding exit point to the radial mode},
  year         = 2019
}
```
