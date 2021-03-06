""" Set of functions to fast pipeline creation. """''
import numpy as np

from ..batchflow import Pipeline, B, V
from ..batchflow.models.tf import TFModel # pylint: disable=no-name-in-module, import-error


def create_preprocess_pipeline(n_samples, sampler=None):
    """Create pipeline for preprocess model.

    Parameters
    ----------
    n_samples : int
        Size of a sample.
    sampler : function or named expression
        Method to sample points within [0, 1] range.
        If callable, it should only take `size` as an argument.

    Returns
    -------
        : Batchflow pipeline
    """
    if sampler is None:
        sampler = np.random.random
    pipeline = (Pipeline()
                .load(fmt='npz')
                .drop_negative(src=['time', 'derivative'])
                .to_log10(src=['time', 'derivative', 'target'])
                .normalize(src=['time', 'derivative', 'target'],
                           src_range=[None, None, 'derivative_q'],
                           dst_range=[None, 'derivative_q', None])
                .get_samples(n_samples, sampler=sampler,
                             src=['time', 'derivative'])
                .make_points(src=['time', 'derivative'], dst=['points'])
                .make_target(src='target')
               )
    return pipeline

def create_train_pipeline(model, model_config, prep=None, **kwargs):
    """Create pipeline for model training.

    Parameters
    ----------
    model : Tensorflow model
        Model for training.
    model_config : dict
        Dict with model parameters.
    prep : None or Pipeline
        If None then preprocessing pipeline will be created with default patameters.
        (n_samples = 100, sampler=np.random.random)
    kwargs : dict
        Parameters for preprocessing pipeline.

    Returns
        : Batchflow pipeline.
    """
    if prep is None:
        n_samples = kwargs.get('n_samples', 100)
        sampler = kwargs.get('sampler', None)
        prep = create_preprocess_pipeline(n_samples, sampler)

    pipeline = prep + (Pipeline()
                       .init_variable('loss', init_on_each_run=list)
                       .init_model('dynamic', model, 'model', config=model_config)
                       .train_model('model', fetches='loss', feed_dict={'points': B('points'),
                                                                        'targets': B('target')},
                                    save_to=V('loss'), mode='a')
                      )
    return pipeline

def create_predict_pipeline(prep=None, load_model=None, **kwargs):
    """Create pipeline for model prediction.

    Parameters
    ----------
    prep : None or Pipeline
        If None then preprocessing pipeline will be created with default patameters.
        (n_samples = 100, sampler=np.random.random)

    load_model : str or Pipeline
        if 'str' than model will be loaded from following path.
        else model will be loaded from given pipeline.
    kwargs : dict
        Parameters for preprocessing pipeline.

    Returns
    -------
        : Batchflow pipeline.
    """
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
                 .clip_values(src=['predictions'])
                 .denormalize(src=['predictions', 'target'],
                              dst=['denorm_predictions', 'denorm_target'],
                              src_range=['derivative_q', 'derivative_q'])
                 .update_variable('predictions', B('denorm_predictions'), mode='e')
                 .update_variable('targets', B('denorm_target'), mode='e')
                )
               )
    return pipeline
