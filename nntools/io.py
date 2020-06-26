import os.path
import nibabel as nb
import numpy as np

# TODO:
# - handle npy files (pickled numpy arrays)
# - Reader / Writer base classes for <Specific>Reader / <Specific>Writer
# - SurfaceReader / SurfaceWriter
# - Read to GPU (can be done with torch I think, possible with TF?)


class VolumeReader:
    """Versatile reader for volume files or objects.
    """

    def __init__(self, dtype=None, allow_memmap=True, copy=True, order=None):
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
        """
        self.dtype = dtype
        self.allow_memmap = allow_memmap
        self.copy = copy
        self.order = order

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
            fname = x
            info['dir'] = os.path.dirname(fname)
            fname = os.path.basename(fname)
            fname, ext = os.path.splitext(fname)
            if ext == '.gz':
                fname, ext0 = os.path.splitext(fname)
                ext = ext0 + ext
            info['basename'] = fname
            info['ext'] = ext
            x = nb.load(x)
        # Then, if nibabel object -> extract affine / header / extra
        if isinstance(x, nb.spatialimages.SpatialImage):
            info['dtype'] = x.get_data_dtype()
            info['affine'] = x.affine
            info['header'] = x.header
            info['extra'] = x.extra
            info['shape'] = x.header.get_data_shape()
        return info

    def read(self, x, dtype=None, allow_memmap=None, copy=None, order=None):
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
        if dtype is None:
            dtype = self.dtype
        dtype = np.dtype(dtype)
        if allow_memmap is None:
            allow_memmap = self.allow_memmap
        if copy is None:
            copy = self.copy
        if order is None:
            order = self.order
        info = {
            'basename': None,
            'dir': None,
            'ext': None,
            'dtype': None,
            'affine': None,
            'header': None,
            'extra': None,
        }
        # If str -> split path into directory / basename / extension
        if isinstance(x, str):
            fname = x
            info['dir'] = os.path.dirname(fname)
            fname = os.path.basename(fname)
            fname, ext = os.path.splitext(fname)
            if ext == '.gz':
                fname, ext0 = os.path.splitext(fname)
                ext = ext0 + ext
            info['basename'] = fname
            info['ext'] = ext
            x = nb.load(x)
        # Then, if nibabel object -> extract affine / header / extra
        if isinstance(x, nb.spatialimages.SpatialImage):
            info['dtype'] = x.get_data_dtype()
            info['affine'] = x.affine
            info['header'] = x.header
            info['extra'] = x.extra
            x = x.get_fdata(dtype=dtype)
        # Then, if not numpy array -> convert to numpy array
        x = np.array(x, copy=copy, order=order)
        if not isinstance(x, np.ndarray):
            raise TypeError("Input type '{}' not handled".format(type(x)))
        # Then, if memory-mapped and not allow_memmap -> load
        if isinstance(x, np.memmap) and not allow_memmap:
            x = x[...]
        return x, info


class VolumeWriter:
    """Versatile writer for volume files or objects."""

    def __init__(self, affine=None, header=None, extra=None,
                  dtype=None, dir=None, ext=None, prefix=None,
                  basename=None, fname=None, dummy=False):
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
        if dummy is None:
            dummy = self.dummy
        if dummy:
            return x
        # --- Parse options ---
        if affine is None:
            affine = self.affine
        if header is None:
            header = self.header
        if extra is None:
            extra = self.extra
        if dtype is None:
            dtype = self.dtype
        if dir is None:
            dir = self.dir
        if ext is None:
            ext = self.ext
        if prefix is None:
            prefix = self.prefix
        if basename is None:
            basename = self.basename
        if fname is None:
            fname = self.fname
        # --- Use input info ---
        if info is None:
            info = {}
        if affine is None:
            affine = info.get('affine')
        if header is None:
            header = info.get('header')
        if extra is None:
            extra = info.get('extra')
        if dtype is None:
            dtype = info.get('dtype')
        if dir is None:
            dir = info.get('dir')
        if ext is None:
            ext = info.get('ext')
        if basename is None:
            basename = info.get('basename')
        # --- Default values if everything is None ---
        if ext is None:
            ext = '.nii.gz'
        if dir is None:
            dir = '.'
        if prefix is None:
            prefix = ''
        if dtype is None:
            dtype = x.dtype
        if basename is None:
            if len(prefix) == 0:
                basename = 'array'
            else:
                basename = ''
        # --- Build full file name ---
        if fname is None:
            fname = prefix + basename + ext
            fname = os.path.join(dir, fname)

        # Some formats do not like 4D volumes, even if the fourth
        # dimension is a singleton. In this case, we remove the fourth
        # dimension. However, if the fourth dimension is > 1, we let it
        # untouched in order to trigger warnings or errors.
        if len(x.shape) > 3 and np.all(np.array(x.shape[3:]) == 1):
            x = x.reshape(x.shape[:3])

        # Build nibabel object
        dtype = np.dtype(dtype)
        onib = nb.spatialimages.SpatialImage(x, affine, header, extra)
        onib.header.set_data_dtype(dtype)
        onib.header.set_data_shape(x.shape)

        # Save on disk
        nb.save(onib, fname)
        return onib


