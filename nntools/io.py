import os.path
import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialImage
from .utils import argpad, argdef

# TODO:
# - Reader / Writer base classes for <Specific>Reader / <Specific>Writer
# - SurfaceReader / SurfaceWriter
# - Read to GPU (can be done with torch I think, possible with TF?)
# - Split base readers (numpy/nibabel/...) into subfunctions or subclasses
#   and add function to try different r/w based on the file extension.
# - use mmap_mode rather than allow_memmmap


def _default_affine(shape):
    """Create default orientation matrix.

    We follow the same convention as nibabel/SPM: (0,0,0) is in the
    center of the field-of-view.

    """
    dim = len(shape)
    shape = np.asarray(shape)
    shift = -shape.astype(np.float64)/2 + 0.5
    mat = np.eye(dim+1, dtype=np.float64)
    mat[:dim, dim] = shift
    return mat


def _fileparts(fname):
    """Split a filename into directory / basename / extension.

    If the last extension is ``.gz``, this function checks if another
    extension is present, in which case it returns ``.<ext>.gz``
    """
    dir = os.path.dirname(fname)
    basename = os.path.basename(fname)
    basename, ext = os.path.splitext(basename)
    if ext == '.gz':
        basename, ext0 = os.path.splitext(basename)
        ext = ext0 + ext
    return dir, basename, ext


class VolumeReader:
    """Versatile reader for volume files or objects."""

    def __init__(self, dtype=None, allow_memmap=True, copy=True,
                 order=None, allow_pickle=False):
        """

        Parameters
        ----------
        dtype : type or str, default=None
            Data type in which to load the input array.

        allow_memmap : bool, default=True
            Allow ``np.memmap`` arrays (i.e., memory-mapped arrays)
            to be returned.

        copy : bool, default=True
            Force the input object to be copied, even if it is already
            an array of the right data type.

        order : {'K', 'A', 'C', 'F'}, default=None
            Specify a specific memory layout.

        allow_pickle : bool, default=False
            Allow loading pickled object arrays stored in npy files.
            Reasons for disallowing pickles include security, as
            loading pickled data can execute arbitrary code. If pickles
            are disallowed, loading object arrays will fail.
        """
        self.dtype = dtype
        self.allow_memmap = allow_memmap
        self.copy = copy
        self.order = order
        self.allow_pickle = allow_pickle

    def __call__(self, *args, **kwargs):
        return self.read(*args, **kwargs)

    def inspect(self, x):
        info = {
            'basename': None,
            'dir': None,
            'ext': None,
            'dtype': None,
            'shape': None,
            'affine': None,
            'header': None,
            'extra': None,
        }
        # If str -> split path into directory / basename / extension
        if isinstance(x, str):
            path, basename, ext = _fileparts(x)
            info['dir'] = path
            info['basename'] = basename
            info['ext'] = ext

            # Memory map data
            if ext in ('.npy', '.npz'):
                # numpy
                x = np.load(x, allow_pickle=self.allow_pickle,
                            mmap_mode='r')
            else:
                # nibabel
                x = nb.load(x)

        # Then, if nibabel object -> extract affine / header / extra
        if isinstance(x, SpatialImage):
            info['dtype'] = x.get_data_dtype()
            info['affine'] = x.affine
            info['header'] = x.header
            info['extra'] = x.extra
            info['shape'] = x.header.get_data_shape()
        elif isinstance(x, np.ndarray):
            info['dtype'] = x.dtype
            info['shape'] = argpad(x.shape, 3, 1)
            info['affine'] = _default_affine(info['shape'])
        return info

    def read(self, x, dtype=None, allow_memmap=None, copy=None, order=None,
             read_info=True):
        """Load (and convert) data stored in a file or array.

        Parameters
        ----------
        x : str or nib.SpatialImage or array_like
            An input array, on disk or in memory.

        dtype : type or str, default=self.dtype
            Data type in which to load the input array.

        allow_memmap : bool, default=self.allow_memmap
            Allow ``np.memmap`` arrays (i.e., memory-mapped arrays)
            to be returned.

        copy : bool, default=self.copy
            Force the input object to be copied, even if it is already
            an array of the right data type.

        order : {'K', 'A', 'C', 'F'}, default=self.order
            Specify a specific memory layout.

        Returns
        -------
        x : np.ndarray
            A numpy array with the specified dtype and memory layout.

        """
        dtype = np.dtype(argdef(dtype, self.dtype))
        allow_memmap = argdef(allow_memmap, self.allow_memmap)
        copy = argdef(copy, self.copy)
        order = argdef(order, self.order)

        if read_info:
            info = dict()

        # If str -> split path into directory / basename / extension
        if isinstance(x, str):
            path, basename, ext = _fileparts(x)
            if read_info:
                info['dir'] = path
                info['basename'] = basename
                info['ext'] = ext

            if ext in ('.npy', '.npz'):
                x = np.load(x, allow_pickle=self.allow_pickle,
                            mmap_mode='r' if self.allow_memmap else None)
            else:
                x = nb.load(x)

        # Then, if nibabel object -> extract affine / header / extra
        if isinstance(x, SpatialImage):
            if read_info:
                info['dtype'] = x.get_data_dtype()
                info['affine'] = x.affine
                info['header'] = x.header
                info['extra'] = x.extra
            x = x.get_fdata(dtype=dtype)
        elif isinstance(x, np.ndarray):
            if read_info:
                info['dtype'] = x.dtype
                info['shape'] = argpad(x.shape, 3, 1)
                info['affine'] = _default_affine(info['shape'])

        # Then, if not numpy array -> convert to numpy array
        x = np.array(x, copy=copy, order=order, dtype=dtype)
        if not isinstance(x, np.ndarray):
            raise TypeError("Input type '{}' not handled".format(type(x)))
        # Then, if memory-mapped and not allow_memmap -> load
        if isinstance(x, np.memmap) and not allow_memmap:
            x = x[...]

        if read_info:
            return x, info
        else:
            return x


class VolumeWriter:
    """Versatile writer for volume files or objects."""

    def __init__(self, affine=None, header=None, extra=None,
                  dtype=None, dir=None, ext=None, prefix=None,
                  basename=None, fname=None, dummy=False):
        """

        Parameters
        ----------
        affine : matrix_like, optional
            Orientation matrix

        header :
            Nibabel header

        extra
            Nibabel extra metadata

        dtype : str or type, optional
            Output data type

        dir : str, default=same as input or current directory
            Output directory

        ext : str, default=same as input or '.nii.gz'
            Output extension

        prefix : str, optional
            Output filename prefix

        basename : str, default=prefixed input or prefix or 'array'
            Output basename

        fname : str
            Output file name (full path + name + extension).
            Default: built from dir/prefix/basename/ext

        dummy : bool, default=False
            Do not write anything

        """
        self.affine = affine
        self.header = header
        self.extra = extra
        self.dtype = dtype
        self.dir = dir
        self.ext = ext
        self.prefix = prefix
        self.basename = basename
        self.fname = fname
        self.dummy = dummy

    def __call__(self, *args, **kwargs):
        return self.write(*args, **kwargs)

    def write(self, x, fname=None, info=None, affine=None, header=None,
              extra=None, dtype=None, dir=None, ext=None, prefix=None,
              basename=None, dummy=None):

        # --- If dummy, do not do anything ---
        dummy = argdef(dummy, self.dummy)
        if dummy:
            return x

        # --- Parse options ---
        # Priority is: function argument / attributes / defaults
        info = argdef(info, {})
        affine = argdef(affine, self.affine, info.get('affine'))
        header = argdef(header, self.header, info.get('header'))
        extra = argdef(extra, self.extra, info.get('extra'))
        dtype = np.dtype(argdef(dtype, self.dtype, info.get('dtype'), x.dtype))
        dir = argdef(dir, self.dir, info.get('dir'), '.')
        ext = argdef(ext, self.ext, info.get('ext'), '.nii.gz')
        prefix = argdef(prefix, self.prefix, info.get('prefix'), '')
        basename = argdef(basename, self.basename, info.get('basename'),
                          'array' if len(prefix) == 0 else '')
        fname = argdef(fname, self.fname,
                       os.path.join(dir, prefix + basename + ext))

        if ext in ('.npy', '.npz'):
            # --- Save using numpy ---
            np.save(fname, x.astype(dtype), allow_pickle=False)
            obj = np.load(fname, allow_pickle=False, mmap_mode='r')

        else:
            # --- Save using nibabel ---

            # Some formats do not like 4D volumes, even if the fourth
            # dimension is a singleton. In this case, we remove the fourth
            # dimension. However, if the fourth dimension is > 1, we let it
            # untouched in order to trigger warnings or errors.
            if len(x.shape) > 3 and np.all(np.array(x.shape[3:]) == 1):
                x = x.reshape(x.shape[:3])

            # Build nibabel object
            obj = SpatialImage(x.astype(dtype), affine, header, extra)
            obj.header.set_data_dtype(dtype)
            obj.header.set_data_shape(x.shape)

            # Save on disk
            nb.save(obj, fname)

        return obj


