import os.path
from glob import glob
from collections import Counter
import numpy as np
import re

# This bit gives some flexibility for the input files:
# . unix-style ~ (HOME) is always expanded
# . unix-style tokens are always expanded
# . if an input file is a directories, we look for all image files
#   within. A list of supported format is specified below. In
#  theory, we could use all extensions supported by nibabel.
default_extensions = ['.nii', '.nii.gz', '.mgh', '.mgz']


def _expansion_pattern(ext=None):
    if ext is None:
        ext = default_extensions
    expansion_pattern = '*['
    for e in ext[:-1]:
        expansion_pattern += e + '|'
    expansion_pattern += ext[-1] + ']'
    return expansion_pattern


def expand_images(images, ext=None):
    """Expand unix token in paths and find specific file types.

    Parameters
    ----------
    images : iterable[str]
        List of paths that can contain tokens.
    ext : list[str], default=['.nii', '.nii.gz', '.mgh', '.mgz']
        File types that are automatically searched for.

    Returns
    -------
    images : list[str]
        List of full paths to individual image files.

    """
    if ext is None:
        ext = default_extensions
    expansion_pattern = _expansion_pattern(ext)
    oimages = []
    for entry in images:
        # entry is: image or dir or token-images or token-dirs
        entry = sorted(glob(os.path.expanduser(entry)))
        for subentry in entry:
            if os.path.isdir(subentry):
                # expand + sort
                subentry = glob(
                    os.path.join(subentry, expansion_pattern)).sort()
                oimages += subentry
            else:
                oimages += [subentry]
    return oimages


def argpad(arg, n, default=None):
    """Pad/crop list so that its length is ``n``.

    Parameters
    ----------
    arg : scalar or iterable
        Input argument(s)
    n : int
        Target length
    default : optional
        Default value to pad with. By fefault, replicate the last value

    Returns
    -------
    arg : list
        Output arguments

    """
    try:
        arg = list(arg)[:n]
    except TypeError:
        arg = [arg]
    if default is None:
        default = arg[-1]
    arg += [default] * max(0, n - len(arg))
    return arg


def argdef(*args):
    """Return the first non-None value from a list of arguments.

    Parameters
    ----------
    value0
        First potential value. If None, try value 1
    value1
        Second potential value. If None, try value 2
    ...
    valueN
        Last potential value

    Returns
    -------
    value
        First non-None value

    """
    args = list(args)
    arg = args.pop(0)
    while arg is None and len(args) > 0:
        arg = args.pop(0)
    return arg


def sub2ind(subs, shape):
    """Convert sub indices (i, j, k) into linear indices.

    Parameters
    ----------
    subs : iterable of array_like
        List of sub-indices. Its length is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : iterable
        Size of each dimension. Its length should be the same as the
        length of ``subs``.

    Returns
    -------
    ind : np.array
        Linear indices

    """
    dim = len(shape)
    if isinstance(subs, np.ndarray) and subs.shape[-1] == dim:
        subs = [subs[..., d] for d in range(dim)]
    ind = np.zeros_like(subs[0])
    # The rightmost dimension is the most rapidly changing one
    # -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]
    stride = np.cumprod(shape[:0:-1])[::-1].tolist() + [1]
    for i, s in zip(subs, stride):
        ind += np.asarray(i) * s
    return ind


def majority(x):
    """Return majority element in a list.

    Parameters
    ----------
    x : iterable
        Input list of elements

    Returns
    -------
    elem
        Majority element

    """
    count = Counter(x)
    max_count = 0
    maj_val = None
    for (key, val) in count.items():
        if val > max_count:
            max_count = val
            maj_val = key
    return maj_val
