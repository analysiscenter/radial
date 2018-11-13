""" Tests for RadialBatch functionality. """

import os
import numpy as np
import pytest

from radial import RadialBatch, dataset as ds
from radial.dataset import R


@pytest.fixture(scope="module")
def setup_batch(request):
    """
    Fixture to setup module. Performs check for presence of test files,
    creates initial index object.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    files = ["rr_1.npz", "sg_1.npz"]

    if np.all([os.path.isfile(os.path.join(path, file)) for file in files]):
        ind = ds.FilesIndex(path=os.path.join(path, '*.npz'), sort=True)
        batch = RadialBatch(ind)
    else:
        raise FileNotFoundError("Test files not found in 'tests/data/'!")

    def teardown_batch():
        """
        Teardown module
        """

    request.addfinalizer(teardown_batch)
    return batch, path

@pytest.fixture(scope="module")
def setup_batch_loaded(request, setup_batch): #pylint: disable=redefined-outer-name
    """
    Fixture to setup module. Creates initial batch object.
    """
    batch = setup_batch[0]
    batch = batch.load(fmt="npz", components=["time", "derivative", "rig_type", "target"])

    def teardown_batch_loaded():
        """
        Teardown module
        """

    request.addfinalizer(teardown_batch_loaded)
    return batch

def test_load(setup_batch): #pylint: disable=redefined-outer-name
    """
    Testing wfdb loader.
    """
    # Arrange
    batch = setup_batch[0]
    # Act
    batch = batch.load(fmt="npz", components=["time", "derivative", "rig_type", "target"])
    # Assert
    assert isinstance(batch.time, np.ndarray)
    assert isinstance(batch.derivative, np.ndarray)
    assert isinstance(batch.rig_type, np.ndarray)
    assert isinstance(batch.target, np.ndarray)
    assert batch.time.shape == (2,)
    assert batch.rig_type.shape == (2,)
    assert batch.target.shape == (2,)
    assert isinstance(batch.derivative[0], np.ndarray)
    assert isinstance(batch.rig_type[0], str)
    del batch

def test_filter_negative(setup_batch_loaded): #pylint: disable=redefined-outer-name
    """
    Test method for filter of negative values.
    """
    # Arrange
    batch = setup_batch_loaded
    # Act
    batch = batch.filter_negative()
    # Assert
    assert np.all(batch.derivative[0] > 0)
    assert np.all(batch.derivative[1] > 0)
    del batch

@pytest.mark.xfail
def test_filter_negative_error(setup_batch_loaded): #pylint: disable=redefined-outer-name
    """
    Test method for error in filtering.
    """
    # Arrange
    batch = setup_batch_loaded
    batch.derivative[0] -= (np.max(batch.derivative[0]) + 1)
    # Act
    batch = batch.filter_negative()
    # Assert

    del batch

def test_get_samples(setup_batch_loaded): #pylint: disable=redefined-outer-name
    """
    Test methods for sampling with interpolation.
    """
    # Arrange
    batch = setup_batch_loaded
    # Act
    batch = batch.get_samples(100, 2, R('beta', a=1, b=1))
    # Assert
    assert batch.time.shape == (2,)
    assert batch.derivative[0].shape == (2, 100)
    assert batch.time[0].shape == (2, 100)
    assert np.all(np.diff(batch.time[0][0]) >= 0)
    assert batch.time.shape == batch.target.shape

def test_unstack_samples(setup_batch_loaded): #pylint: disable=redefined-outer-name
    """
    Test method for unstacking samples with generation of
    new batch.
    """
    # Arrange
    batch = setup_batch_loaded
    # Act
    batch = (batch
             .get_samples(100, 2, R('beta', a=1, b=1))
             .unstack_samples()
            )
    # Assert
    assert batch.time.shape == (4,)
    assert batch.derivative[0].shape == (100,)
    assert np.all(np.diff(batch.time[0]) >= 0)
    assert batch.time.shape == batch.target.shape
