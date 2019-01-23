""" Pipelines for radial regime regression. """

from functools import wraps

import numpy as np
import scipy as sc

from . import radial_batch_tools as bt
from ..batchflow import Batch, action, inbatch_parallel, any_action_failed, FilesIndex, DatasetIndex, R, B, V, C, Dataset, Pipeline

from ..batchflow.models.tf import TFModel, VGG7
from . import RadialBatch, RadialImagesBatch


def log(*args):
    return np.array(list(map(np.log10, args)))


def create_load_ppl(image_index=None):
    load_ppl = (Pipeline()
                    .load(fmt='npz')
                    .drop_negative()
                    .apply_transform(log, src=['time', 'derivative', 'target'], dst=['log_time', 'log_derivative', 'log_target'])
                    .normalize(src=['log_time', 'log_derivative'], dst=['log_norm_time', 'log_norm_derivative'],\
                               dst_range=[None, 'derivative_range'])
                    .normalize(src='log_target', dst='log_norm_target', src_range='derivative_range')
                    .load(fmt='image', src=image_index, components='images')
                    .resize((300, 200))
                    .crop(origin='center', shape=(270, 180))
                    .to_array()
                    .multiply(1/255.)
                    .load(fmt='csv', src='january/targets.csv', components='target', index_col='index')
                    .expand_dims(src='log_norm_target')
                    .expand_dims(src='target')
               )
    return load_ppl