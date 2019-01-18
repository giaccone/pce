# ==============
# import modules
# ==============
import numpy as np

# =========
# functions
# =========
def human_readable_time(time):
    """
    HUMAN_READABLE_TIME convert a time ns, us, ms depending on its value

    Paramaters
    ----------
    time (float) :
        time value
    
    Returns
    -------
    time (float):
        time value converted
    units (str):
        unit of measurement

    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 11.08.2018
    HISTORY:
    """
    if time < 1e-9:
        time = time * 1e6
        unit = 'ns'
    elif time < 1e-6:
        time = time * 1e6
        unit = 'us'
    elif time < 1e-3:
        time = time * 1e3
        unit = 'ms'
    else:
        unit = 's'

    return time, unit


def columnize(*args):
    """
    COLUMNIZE convert a list of arrays into a list of column numpy arrays
    
    Parameters
    ----------
    *args:
        variable length argument list including all arrays
    
    Returns
    -------
    column_arrays (list):
        list of numpy column arrays

    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 14.08.2018
    HISTORY:
    """

    column_arrays = []
    for ele in args:
        if isinstance(ele, (list, tuple)):
            ele = np.array(ele)
        ele.shape = (-1, 1)
        column_arrays.append(ele)

    return column_arrays


def axis_equal_3d(fig_handle):
    """
    AXIS_EQUAL_3D sets the three axis with the same aspect ratio
    
    Parameters
    ----------
    fig_handle (Figure) :
        handle to existing figure
    

    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 13.08.2018
    REF: https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
    HISTORY:
    """

    axis_handle = fig_handle.gca(projection='3d')
    axis_handle.axis('tight')

    extents = np.array([getattr(axis_handle, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(axis_handle, 'set_{}lim'.format(dim))(ctr - r, ctr + r)