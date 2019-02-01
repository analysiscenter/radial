""" Set of functions to fast pipeline creation. """''
import numpy as np

from ..batchflow import Pipeline, B, V


def create_preprocess_pipeline(n_samples, sampler=None):
    """ create pipeline for preprocess model"""
    if sampler is None:
        sampler = np.random.random
    pipeline = (Pipeline()
                .load(fmt='npz')
                .drop_negative(src=['time', 'derivative'])
                .normalize(src=['time', 'derivative', 'target'],
                           dst=['time', 'derivative', 'target'],
                           src_range=[None, None, 'derivative_q'],
                           dst_range=[None, 'derivative_q', None])
                .get_samples(n_samples, n_samples=1, sampler=sampler, src=['time', 'derivative'])
                .make_points(src=['time', 'derivative'], dst=['points'])
                .make_target(src='target')
               )
    return pipeline

def create_train_pipeline(model, model_config, prep=None, **kwargs):
    """crete pipeline for model training """
    if prep is None:
        n_samples = kwargs.get('n_samples', 50)
        sampler = kwargs.get('sampler', None)
        prep = create_preprocess_pipeline(n_samples, sampler)
    pipeline = prep + (Pipeline()
                       .init_variable('loss', init_on_each_run=list)
                       .init_model('dynamic', model, 'model', config=model_config)
                       .train_model('model', fetches='loss', feed_dict={'points': B('points'),
                                                                        'targets': B('target')},
                                    save_to=V('loss'), mode='w')
                      )
    return pipeline
