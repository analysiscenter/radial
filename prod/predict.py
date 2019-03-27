"""File with function that takes log10 time and log10 of the derivative of the pressure and reutrns log10 point of exit to radial mode.

BASIC USAGE:
foo@bar:~$ python predict.py -p ./path/to/data.npy -m /path/to/model (optional)

or just run python predict.py --help

"""
import os
import sys

import argparse
import numpy as np

sys.path.insert(0, os.path.join('..'))

from radial.core import RadialBatch
from radial.batchflow.models.tf import TFModel
from radial.batchflow import Dataset, Pipeline, B

def make_prediction():
    """load data from path, given with argument `-p`.
    This data should be a NPY file with 2d numpy array with shape = (2, N),
    where N is lenght of time and derivative of the pressure changes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="Path to file with time and derivative of the pressure.",
                        required=True)
    parser.add_argument('-m', '--model', type=str, help="Path to saved model. Default = ./saved_model", default='./saved_model')
    args = parser.parse_args()
    data = np.load(args.path)
    model_path = args.model
    return predict(data[0], data[1], model_path)

def predict(time, derivative, model_path):
    """Make a prediction using loaded model

    Parameters
    ----------
    time : numpy array
        log 10 time values
    derivative : numpy array
        log 10 derivative values

    Returns
    -------
        : radial mode point
    """
    time = np.array([time] + [None])[:-1]
    derivative = np.array([derivative] + [None])[:-1]

    ds = Dataset(index=1, batch_class=RadialBatch)
    prep_pipeline = (Pipeline()
                     .load(src=(time, derivative), components=['time', 'derivative'])
                     .drop_negative(src=['time', 'derivative'])
                     .drop_outliers(src=['time', 'derivative'])
                     .normalize(src=['time', 'derivative'],
                                dst_range=[None, 'derivative_q'])
                     .get_samples(100, sampler=np.random.random, src=['time', 'derivative'])
                     .make_points(src=['time', 'derivative'], dst=['points'])
                     )
    test_pipeline = prep_pipeline + (Pipeline()
                                     .init_variable('predictions', init_on_each_run=list)
                                     .init_model('dynamic', TFModel, 'model',
                                                 config={'load' : {'path' : model_path},
                                                         'build': False})
                                     .init_variable('ind', init_on_each_run=list)
                                     .update_variable('ind', B('indices'), mode='e')
                                     .predict_model('model', fetches='predictions',
                                                    feed_dict={'points': B('points')},
                                                    save_to=B('predictions'), mode='w')
                                     .clip_values(src=['predictions'])
                                     .denormalize(src=['predictions'],
                                                  src_range=['derivative_q'])
                                     .update_variable('predictions', B('predictions'), mode='e')
                                     ) << ds
    test_pipeline.run(1, n_epochs=10)
    return np.mean(np.array(test_pipeline.get_variable('predictions')))

if __name__ == "__main__":
    sys.exit(make_prediction())
