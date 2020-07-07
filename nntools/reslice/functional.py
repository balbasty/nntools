"""Tools for reslicing volumes implemented in a Functional paradigm."""

from .object import Reslicer, ReslicerLike, Resizer, Upsampler, Downsampler, \
               ShapeResizer, VoxelResizer
from ..hints import Matrix, Vector, AnyArray
from typing import Mapping


def reslice(x, output_affine=None, output_shape=None, input_affine=None,
            **kwargs):
    # type: (AnyArray, Matrix, Vector, Matrix, Mapping) -> AnyArray
    """Reslice a volume to another space (= affine + shape).

    Parameters
    ----------
    x : file_like or array_like
        Input volume

    input_affine : matrix_like, default=read from file
        Input orientation matrix, mapping voxels to world space

    output_shape : vector_like, default=same as input
        Output (3D spatial) shape

    output_affine : matrix_like, default=same as input
        Output orientation matrix, mapping voxels to world space

    Other Parameters
    ----------------
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
    reslicer = Reslicer()
    return reslicer(x,
                    output_affine=output_affine,
                    output_shape=output_shape,
                    input_affine=input_affine,
                    **kwargs)


def reslice_like(x, reference_volume=None, input_affine=None,
                 **kwargs):
    # type: (AnyArray, AnyArray, Matrix, Mapping) -> AnyArray
    """Reslice a volume to the space of another volume.

    Parameters
    ----------
    x : file_like or array_like
        Input volume.

    reference_volume : file_like or array_like
        Reference volume, whose shape and affine matrix define the
        reference space

    input_affine : matrix_like, default=guessed from input
        Input orientation matrix, mapping voxels to world space

    Other Parameters
    ----------------
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
    reslicer = ReslicerLike()
    return reslicer(x,
                    reference_volume=reference_volume,
                    input_affine=input_affine,
                    **kwargs)


def resize(x, factor, output_shape=None, **kwargs):
    # type: (AnyArray, Vector, Vector, Mapping) -> AnyArray
    """Resize an image by a given factor (greater or lower than one).

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When upsampling by
    an integer factor or downsampling by a divisor of the input size,
    the bottom right corners should also match.

    Parameters
    ----------
    x : file_like or array_like
        Input volume.

    factor : vector_like
        Factor by which to scale the voxel size

    output_shape : vector_like, default=input shape
        Output shape

    Other Parameters
    ----------------
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
    reslicer = Resizer()
    return reslicer(x,
                    factor=factor,
                    output_shape=output_shape,
                    **kwargs)


def upsample(x, factor=2, output_shape=None, **kwargs):
    # type: (AnyArray, Vector, Vector, Mapping) -> AnyArray
    """Upsample images by a factor.

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When upsampling by
    an integer factor, the bottom right corners should also match.

    Parameters
    ----------
    x : file_like or array_like
        Input volume.

    factor : vector_like, default=2
        Factor by which to scale the voxel size

    output_shape : vector_like, default=input shape
        Output shape

    Other Parameters
    ----------------
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
    reslicer = Upsampler()
    return reslicer(x,
                    factor=factor,
                    output_shape=output_shape,
                    **kwargs)


def downsample(x, factor=2, output_shape=None, **kwargs):
    # type: (AnyArray, Vector, Vector, Mapping) -> AnyArray
    """Downsample images by a factor.

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When downsampling
    by a divisor of the input size, the bottom right corners should
    also match.

    Parameters
    ----------
    x : file_like or array_like
        Input volume.

    factor : vector_like, default=2
        Factor by which to scale the voxel size

    output_shape : vector_like, default=input shape
        Output shape

    Other Parameters
    ----------------
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
    reslicer = Downsampler()
    return reslicer(x,
                    factor=factor,
                    output_shape=output_shape,
                    **kwargs)


def resize_shape(x, output_shape=None, **kwargs):
    # type: (AnyArray, Vector, Mapping) -> AnyArray
    """Resize the shape of an image to match a target shape.

    The top-left and bottom right corners of both field-of-views are
    used as anchors.

    Parameters
    ----------
    x : file_like or array_like
        Input volume.

    output_shape : vector_like, default=input shape
        Output shape

    Other Parameters
    ----------------
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
    reslicer = ShapeResizer()
    return reslicer(x,
                    output_shape=output_shape,
                    **kwargs)


def resize_voxel(x, output_vs=None, **kwargs):
    # type: (AnyArray, Vector, Mapping) -> AnyArray
    """Resize the voxel size of an image to match a target voxel size.

    The top-left and bottom right corners of both field-of-views are
    used as anchors.

    Parameters
    ----------
    x : file_like or array_like
        Input volume.

    output_vs : vector_like, optional
        Output voxel size

    Other Parameters
    ----------------
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
    reslicer = VoxelResizer()
    return reslicer(x,
                    output_vs=output_vs,
                    **kwargs)
