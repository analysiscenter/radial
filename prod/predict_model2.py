"""File with function that takes log10 time and log10 of the derivative of the pressure and reutrns log10 point of exit to radial mode. """
import os
import sys

import argparse
import numpy as np

sys.path.insert(0, os.path.join('..'))

from radial.core import RadialBatch
from radial.batchflow.models.tf import TFModel
from radial.batchflow import Dataset, Pipeline, B, C
from radial.core.radial_batch_tools import *

GRID_SIZE = 200

def make_prediction():
    """load data from path, given with argument `-p`.
    This data should be a NPY file with 2d numpy array with shape = (2, N),
    where N is lenght of time and derivative of the pressure changes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="Path to file with time and derivative of the pressure.",
                        required=True)
    args = parser.parse_args()
    data = np.load(args.path)
    return predict(data[0], data[1])

def predict(time, derivative):
    time = np.array([time] + [None])[:-1]
    derivative = np.array([derivative] + [None])[:-1]

    ds = Dataset(index=1, batch_class=RadialBatch)
    prep_pipeline = (Pipeline()
                        .init_variable('loss_history_dict', init_on_each_run=0)
                        .hard_negative_sampling(statistics_name='loss_history_dict', fraction=0.33)
                        .load(src=(time, derivative), components=['time', 'derivative'])
                        .drop_negative(src=['time', 'derivative'])
                        .apply_transform(log, src=['time', 'derivative'])
                        .normalize(src=['time', 'derivative'], \
                                   dst_range=[None, 'derivative_range'])
                        .make_grid_data(src=['time', 'derivative'], dst=['log_norm_time_grid', 'log_norm_derivative_grid'],
                                        grid_size=GRID_SIZE)
                        .make_array(src='log_norm_derivative_grid', dst='derivative_grid')
    )
    test_pipeline = prep_pipeline + (Pipeline()
                        .init_variable('predictions', init_on_each_run=list)
                        .init_model('dynamic', TFModel, 'model',
                                     config={'load' : {'path' : './../standards/saved_model/'},
                                             'build': False})
                        .predict_model('model', fetches='predictions',
                                       feed_dict={'signals': B('derivative_grid')},
                                       save_to=B('predictions'), mode='w')
                        .clip_values(src='predictions', dst='predictions')
                        .denormalize(src='predictions', dst='denorm_predictions',
                                     src_range='derivative_range')
                        .update_variable('predictions', B('denorm_predictions'), mode='e')
    ) << ds
    test_pipeline.run(1, n_epochs=1)
    return test_pipeline.get_variable('predictions')

if __name__ == "__main__":
    sys.exit(make_prediction())
