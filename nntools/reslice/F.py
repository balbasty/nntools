"""Tools for reslicing volumes implemented in a Functional paradigm."""

import numpy as np
from ..io import isfile, VolumeWriter, VolumeConverter
from .O import Reslicer, ReslicerLike, Resizer, Upsampler, Downsampler, \
               ShapeResizer, VoxelResizer


def _select_writer(x, writer, map_writer, prefix, map_prefix='map_'):
    """Choose writer based on input type."""
    if isfile(x):
        if writer is None:
            writer = VolumeWriter(prefix=prefix)
        if map_writer is None:
            map_writer = VolumeWriter(prefix=map_prefix, dtype=np.float64)
    else:
        x = np.asarray(x)
        if writer is None:
            writer = VolumeConverter(x.dtype)
        if map_writer is None:
            map_writer = VolumeConverter(np.float64)
    return writer, map_writer


def reslice(x,
            output_affine=None, output_shape=None, input_affine=None, *,
            writer=None, map_writer=None, **kwargs):
    """Reslice a volume to another space (= affine + shape).

    Parameters
    ----------
    x : file_like or array_like
        Input volume

    input_affine : matrix_like, default=read from file
        Input orientation matrix, mapping voxels to world space

    output_affine : matrix_like, default=same as input
        Output orientation matrix, mapping voxels to world space

    output_shape : iterable, default=same as input
        Output (3D spatial) shape

    order : int, default=1
        Interpolation order

    bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='mirror'
        Boundary conditions when sampling out-of-bounds

    extrapolate : bool, default=False
        Use boundary conditions to extrapolate data outside of
        the original field-of-view.

    compute_map : bool, default=False
        Compute reliability map

    writer : io.VolumeWriter, optional
        Writer object for the resliced image

    map_writer : io.VolumeWriter, optional
        Writer object for the reliability map

    Returns
    -------
    x : file_like or ndarray
        Resliced volume

    map : file_like or ndarray, optional
        Reliability map

    """
    # Choose appropriate writer
    writer, map_writer = _select_writer(x, writer, map_writer, 'resliced_')

    # Reslice
    reslicer = Reslicer(writer=writer, map_writer=map_writer)
    return reslicer(x,
                    output_affine=output_affine,
                    output_shape=output_shape,
                    input_affine=input_affine,
                    **kwargs)


def reslice_like(x, reference_volume=None, input_affine=None, *,
                 writer=None, map_writer=None, **kwargs):
    """Reslice a volume to the space of another volume.

    Parameters
    ----------
    x : str or array_like
        Input volume.

    reference_volume : file_like or array_like
        Reference volume, whose shape and affine matrix define the
        reference space

    input_affine : matrix_like, default=guessed from input
        Input orientation matrix, mapping voxels to world space

    order : int, default=1
        Interpolation order

    bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='mirror'
        Boundary conditions when sampling out-of-bounds

    compute_map : bool, default=False
        Compute reliability map

    writer : io.VolumeWriter, optional
        Writer object for the resliced image

    map_writer : io.VolumeWriter, optional
        Writer object for the reliability map

    Returns
    -------
    y : ndarray
        Resliced volume

    map : file_like or ndarray, optional
        Reliability map

    """
    # Choose appropriate writer
    writer, map_writer = _select_writer(x, writer, map_writer, 'resliced_')

    # Reslice
    reslicer = ReslicerLike(writer=writer, map_writer=map_writer)
    return reslicer(x,
                    reference_volume=reference_volume,
                    input_affine=input_affine,
                    **kwargs)


def resize(x, factor, output_shape=None, *, writer=None, map_writer=None,
           **kwargs):
    """Resize an image by a given factor (greater or lower than one).

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When upsampling by
    an integer factor or downsampling by a divisor of the input size,
    the bottom right corners should also match.

    Parameters
    ----------
    x : str or array_like
        Input volume.

    factor : iterable
        Factor by which to scale the voxel size

    output_shape : iterable, default=input shape
        Output shape

    order : int, default=1
        Interpolation order

    bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='mirror'
        Boundary conditions when sampling out-of-bounds

    compute_map : bool, default=False
        Compute reliability map

    writer : io.VolumeWriter, optional
        Writer object for the resliced image

    map_writer : io.VolumeWriter, optional
        Writer object for the reliability map

    Returns
    -------
    y : ndarray
        Resized volume

    map : file_like or ndarray, optional
        Reliability map

    """
    # Choose appropriate writer
    writer, map_writer = _select_writer(x, writer, map_writer, 'resized_')

    # Reslice
    reslicer = Resizer(writer=writer, map_writer=map_writer)
    return reslicer(x,
                    factor=factor,
                    output_shape=output_shape,
                    **kwargs)


def upsample(x, factor=2, output_shape=None, *, writer=None, map_writer=None,
             **kwargs):
    """Upsample images by a factor.

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When upsampling by
    an integer factor, the bottom right corners should also match.

    Parameters
    ----------
    x : str or array_like
        Input volume.

    factor : iterable, default=2
        Factor by which to scale the voxel size

    output_shape : iterable, default=input shape
        Output shape

    order : int, default=1
        Interpolation order

    bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='mirror'
        Boundary conditions when sampling out-of-bounds

    compute_map : bool, default=False
        Compute reliability map

    writer : io.VolumeWriter, optional
        Writer object for the resliced image

    map_writer : io.VolumeWriter, optional
        Writer object for the reliability map

    Returns
    -------
    y : ndarray
        Resized volume

    map : file_like or ndarray, optional
        Reliability map

    """
    # Choose appropriate writer
    writer, map_writer = _select_writer(x, writer, map_writer, 'upsampled_')

    # Reslice
    reslicer = Upsampler(writer=writer, map_writer=map_writer)
    return reslicer(x,
                    factor=factor,
                    output_shape=output_shape,
                    **kwargs)


def downsample(x, factor=2, output_shape=None, *, writer=None, map_writer=None,
               **kwargs):
    """Downsample images by a factor.

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When downsampling
    by a divisor of the input size, the bottom right corners should
    also match.

    Parameters
    ----------
    x : str or array_like
        Input volume.

    factor : iterable, default=2
        Factor by which to scale the voxel size

    output_shape : iterable, default=input shape
        Output shape

    order : int, default=1
        Interpolation order

    bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='mirror'
        Boundary conditions when sampling out-of-bounds

    compute_map : bool, default=False
        Compute reliability map

    writer : io.VolumeWriter, optional
        Writer object for the resliced image

    map_writer : io.VolumeWriter, optional
        Writer object for the reliability map

    Returns
    -------
    y : ndarray
        Resized volume

    map : file_like or ndarray, optional
        Reliability map

    """
    # Choose appropriate writer
    writer, map_writer = _select_writer(x, writer, map_writer, 'downsampled_')

    # Reslice
    reslicer = Downsampler(writer=writer, map_writer=map_writer)
    return reslicer(x,
                    factor=factor,
                    output_shape=output_shape,
                    **kwargs)


def resize_shape(x, output_shape=None, *, writer=None, map_writer=None,
                 **kwargs):
    """Resize the shape of an image to match a target shape.

    The top-left and bottom right corners of both field-of-views are
    used as anchors.

    Parameters
    ----------
    x : str or array_like
        Input volume.

    output_shape : iterable, default=input shape
        Output shape

    order : int, default=1
        Interpolation order

    bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='mirror'
        Boundary conditions when sampling out-of-bounds

    compute_map : bool, default=False
        Compute reliability map

    writer : io.VolumeWriter, optional
        Writer object for the resliced image

    map_writer : io.VolumeWriter, optional
        Writer object for the reliability map

    Returns
    -------
    y : ndarray
        Resized volume

    map : file_like or ndarray, optional
        Reliability map

    """
    # Choose appropriate writer
    writer, map_writer = _select_writer(x, writer, map_writer, 'resized_')

    # Reslice
    reslicer = ShapeResizer(writer=writer, map_writer=map_writer)
    return reslicer(x,
                    output_shape=output_shape,
                    **kwargs)


def resize_voxel(x, output_vs=None, *, writer=None, map_writer=None,
                 **kwargs):
    """Resize the voxel size of an image to match a target voxel size.

    The top-left and bottom right corners of both field-of-views are
    used as anchors.

    Parameters
    ----------
    x : str or array_like
        Input volume.

    output_vs : iterable, optional
        Output voxel size

    order : int, default=1
        Interpolation order

    bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='mirror'
        Boundary conditions when sampling out-of-bounds

    compute_map : bool, default=False
        Compute reliability map

    writer : io.VolumeWriter, optional
        Writer object for the resliced image

    map_writer : io.VolumeWriter, optional
        Writer object for the reliability map

    Returns
    -------
    y : ndarray
        Resized volume

    map : file_like or ndarray, optional
        Reliability map

    """
    # Choose appropriate writer
    writer, map_writer = _select_writer(x, writer, map_writer, 'rescaled_')

    # Reslice
    reslicer = VoxelResizer(writer=writer, map_writer=map_writer)
    return reslicer(x,
                    output_vs=output_vs,
                    **kwargs)
