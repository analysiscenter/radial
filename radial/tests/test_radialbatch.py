""" Tests for RadialBatch functionality. """

import os
import numpy as np
import pytest

from radial import RadialBatch, dataset as ds


@pytest.fixture(scope="module")
def setup_module_load(request):
    """
    Fixture to setup module. Performs check for presence of test files,
    creates initial batch object.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    files = ["rr_1.npz", "sg_1.npz"]

    if np.all([os.path.isfile(os.path.join(path, file)) for file in files]):
        ind = ds.FilesIndex(path=os.path.join(path, '*.npz'), sort=True)
    else:
        raise FileNotFoundError("Test files not found in 'tests/data/'!")

    def teardown_module_load():
        """
        Teardown module
        """

    request.addfinalizer(teardown_module_load)
    return ind, path

def test_load(setup_module_load): #pylint: disable=redefined-outer-name
    """
    Testing wfdb loader.
    """
    # Arrange
    ind = setup_module_load[0]
    batch = RadialBatch(ind)
    # Act
    batch = batch.load(fmt="npz", components=["time", "pressure", "derivative", "rig_type", "target"])
    # Assert
    assert isinstance(batch.time, np.ndarray)
    assert isinstance(batch.pressure, np.ndarray)
    assert isinstance(batch.derivative, np.ndarray)
    assert isinstance(batch.rig_type, np.ndarray)
    assert isinstance(batch.target, np.ndarray)
    assert batch.time.shape == (2,)
    assert batch.rig_type.shape == (2,)
    assert batch.target.shape == (2,)
    assert isinstance(batch.derivative[0], np.ndarray)
    assert isinstance(batch.pressure[0], np.ndarray)
    assert batch.pressure[1] is None
    assert isinstance(batch.rig_type[0], str)
    del batch
