"""File with decorators."""
from functools import wraps

import numpy as np


def _safe_make_array(dst, length):
    """ Makes array from dst data. Raises exception if length of resulting array
    is not equal to length. If dst is None returns array of None of length length.
    Parameters
    ----------
    dst : str or None or list, tuple, np.ndarray of str
        Contains names of batch component(s) to be processed
    length : int
        Desired length of array of source components
    Returns
    -------
    np.array
    """
    if not isinstance(dst, (list, tuple, np.ndarray)):
        if not dst:
            dst = np.array([None] * length)
        else:
            dst = np.asarray(dst).reshape(-1)
    elif not len(dst) == length:
        raise ValueError('Number of given components must be equal')
    return np.array(dst)

def init_components(method):
    """ Decorator used for preprocessing kwargs such as src, dst, src_range, dst_range.
    Modifies str to list of str, inserts default values and raises ValueError if mandatory
    parameters are missing.

    Parameters
    ----------
    method : method to be decorated

    Returns
    -------
    Method with updated kwargs
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        """ Wrapper
        """
        src = kwargs.get('src', None)
        dst = kwargs.get('dst', None)
        src_range = kwargs.get('src_range', None)
        dst_range = kwargs.get('dst_range', None)
        src = np.asarray(src).reshape(-1)

        len_dst = len(src) if dst is None else len(dst)
        dst = _safe_make_array(dst, len_dst)

        src_range = _safe_make_array(src_range, len(src))
        dst_range = _safe_make_array(dst_range, len(src))

        for i, component in enumerate(src):
            if not component:
                raise ValueError('Required argument `src` (pos 1) not found')
            if isinstance(component, str):
                pass
            if not hasattr(self, component):
                raise ValueError('Component passed in src does not exist')
            if len(dst) > i:
                if not dst[i]:
                    dst[i] = component
                if not hasattr(self, dst[i]):
                    setattr(self, dst[i], self.array_of_nones)

            if src_range[i] and not hasattr(self, src_range[i]):
                raise ValueError('Component provided in src_range does not exist')
            if dst_range[i] and not hasattr(self, dst_range[i]):
                setattr(self, dst_range[i], self.array_of_nones)
        kwargs.update(src=src, dst=dst, src_range=src_range, dst_range=dst_range)
        return method(self, *args, **kwargs)
    return wrapper
