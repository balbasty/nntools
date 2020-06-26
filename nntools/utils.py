import os.path
from glob import glob

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


def argpad(arg, n):
    try:
        arg = list(arg)[:n]
    except TypeError:
        arg = [arg]
    arg += arg[-1:] * max(0, n - len(arg))
    return arg
