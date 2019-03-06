""" Set of functions to fast pipeline creation. """''
import numpy as np

from ..batchflow import Pipeline, B, V
from ..batchflow.models.tf import TFModel # pylint: disable=no-name-in-module, import-error


def create_preprocess_pipeline(n_samples, sampler=None):
    """ create pipeline for preprocess model"""
    if sampler is None:
        sampler = np.random.random
    pipeline = (Pipeline()
                .load(fmt='npz')
                .drop_negative(src=['time', 'derivative'])
                .to_log10(src=['time', 'derivative', 'target'])
                .normalize(src=['time', 'derivative', 'target'],
                           src_range=[None, None, 'derivative_q'],
                           dst_range=[None, 'derivative_q', None])
                .get_samples(n_samples, n_samples=1, sampler=np.random.random,
                             src=['time', 'derivative'])
                .make_points(src=['time', 'derivative'], dst=['points'])
                .make_target(src='target')
                )
    return pipeline

def create_train_pipeline(model, model_config, prep=None, **kwargs):
    """crete pipeline for model training """
    if prep is None:
        n_samples = kwargs.get('n_samples', 100)
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

def create_predict_pipeline(prep=None, load_model=None, **kwargs):
    """create pipeline for model prediction"""
    if prep is None:
        n_samples = kwargs.get('n_samples', 50)
        sampler = kwargs.get('sampler', None)
        prep = create_preprocess_pipeline(n_samples, sampler)

    if load_model is None:
        raise ValueError("`load_model` should be or src or Pipeline not None")
    if isinstance(load_model, str):
        model_pipeline = Pipeline().init_model('dynamic', TFModel, 'model',
                                               config={'load' : {'path' : load_model},
                                                       'build': False})
    else:
        model_pipeline = Pipeline().import_model('model', load_model)

    pipeline = (prep + model_pipeline +
                (Pipeline()
                 .init_variable('predictions', init_on_each_run=list)
                 .init_variable('targets', init_on_each_run=list)
                 .init_variable('ind', init_on_each_run=list)
                 .update_variable('ind', B('indices'), mode='e')
                 .predict_model('model', fetches='predictions',
                                feed_dict={'points': B('points'),
                                           'targets': B('target')},
                                save_to=B('predictions'), mode='w')
                 .denormalize(src=['predictions', 'target'],
                              dst=['denorm_predictions', 'denorm_target'],
                              src_range=['derivative_q', 'derivative_q'])
                 .update_variable('predictions', B('denorm_predictions'), mode='e')
                 .update_variable('targets', B('denorm_target'), mode='e')
                 )
                )
    return pipeline
