"""Tools for reslicing volumes implemented in an Object-Oriented paradigm."""

# WARNING: reslice.F imports reslice.O, so the opposite import is forbidden

from ..io import VolumeReader, VolumeWriter, VolumeConverter, isfile
from ..interpolate import affine_grid, sample_grid, reliability_grid
from ..utils import argpad, argdef
from ..linalg import lmdiv, rmdiv
from ..space.affine import change_layout
from ..hints import Matrix, Vector, AnyArray
from typing import Tuple
import numpy as np
from scipy.ndimage.interpolation import spline_filter1d
from scipy.ndimage import map_coordinates

# TODO:
# - Possibility to read/write features one at a time to save memory.


def _select_writer(x, writer, map_writer, prefix, map_prefix='map_'):
    """Choose writer based on input type."""
    if isfile(x):
        if writer is None:
            writer = VolumeWriter(prefix=prefix)
        if map_writer is None:
            map_writer = VolumeWriter(prefix=map_prefix, dtype=np.float32)
    else:
        x = np.asarray(x)
        if writer is None:
            writer = VolumeConverter(x.dtype)
        if map_writer is None:
            map_writer = VolumeConverter(np.float64)
    return writer, map_writer


class Reslicer:
    """Generic class for reslicing a volume.

    This class uses the input and output affine orientation matrix
    to compute a voxel-to-voxel mapping between two spaces.

    It is use as a base class by most high-level reslicers.
    """

    output_prefix = 'resliced_'
    map_prefix = 'map_'

    def __init__(self, output_affine=None, output_shape=None, *,
                 output_layout=None,
                 order=1, bound='nearest', extrapolate=False,
                 compute_map=False, ensure_multiple=None,
                 writer=None, map_writer=None):
        """

        Parameters
        ----------
        output_affine : matrix_like, default=same as input
            Output orientation matrix, mapping voxels to world space

        output_shape : iterable, default=same as input
            Output (3D spatial) shape

        Other Parameters
        ----------------
        order : int, default=1
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='nearest'
            Boundary conditions when sampling out-of-bounds

        extrapolate : bool, default=False
            Use boundary conditions to extrapolate data outside of
            the original field-of-view.

        compute_map : bool, default=False
            Compute reliability map

        ensure_multiple : vector_like[int], optional
            Ensures that the shape is a multiple of a value (e.g., 32).
            This can be useful for convolutional neural networks.
            If used, pad non-singleton dimensions until they fulfill
            the condition.

        writer : io.VolumeWriter, optional
            Writer object for the resliced image

        map_writer : io.VolumeWriter, optional
            Writer object for the reliability map
        """
        self.output_affine = output_affine
        self.output_shape = output_shape
        self.output_layout = output_layout
        self.order = order
        self.bound = bound
        self.extrapolate = extrapolate
        self.compute_map = compute_map
        self.ensure_multiple = ensure_multiple
        self.reader = VolumeReader()
        self.writer = writer
        self.map_writer = map_writer

    def __call__(self, x,
                 output_affine=None, output_shape=None, input_affine=None, *,
                 output_layout=None,
                 order=None, bound=None, extrapolate=None, compute_map=None,
                 ensure_multiple=None, writer=None, map_writer=None):
        """Reslice a volume to a target shape and orientation (affine matrix).

        Parameters
        ----------
        x : str or array_like
            Input volume.

        output_affine : matrix_like, default=self.output_affine
            Output orientation matrix, mapping voxels to world space

        output_shape : iterable, default=self.output_shape
            Output (3D spatial) shape

        input_affine : matrix_like, default=guessed from input
            Input orientation matrix, mapping voxels to world space

        Other Parameters
        ----------------
        order : int, default=self.order
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar,
                default=self.bound
            Boundary conditions when sampling out-of-bounds

        extrapolate : bool, default=self.extrapolate
            Use boundary conditions to extrapolate data outside of
            the original field-of-view.

        compute_map : bool, default=self.compute_map
            Compute reliability map

        ensure_multiple : vector_like[int], default=self.ensure_multiple
            Ensures that the shape is a multiple of a value (e.g., 32).
            This can be useful for convolutional neural networks.
            If used, pad non-singleton dimensions until they fulfill
            the condition.

        writer : io.VolumeWriter, default=self.writer
            Writer object for the resliced image

        map_writer : io.VolumeWriter, default=self.map_writer
            Writer object for the reliability map

        Returns
        -------
        y : np.ndarray
            Resliced volume

        map : file_like or np.ndarray, optional
            Reliability map

        """

        # Select appropriate writer
        writer, map_writer = _select_writer(x, self.writer,
                                            self.map_writer,
                                            self.output_prefix,
                                            self.map_prefix)

        # Load input volume
        x, info = self.reader(x, dtype=np.float32)

        # Parse options
        input_affine = argdef(input_affine, info.get('affine'), np.eye(4))
        output_affine = argdef(output_affine, self.output_affine, input_affine)
        output_shape = argdef(output_shape, self.output_shape, x.shape)
        output_shape = argpad(output_shape, 3)
        output_layout = argdef(output_layout, self.output_layout)
        order = argdef(order, self.order, 1)
        bound = argdef(bound, self.bound, 'mirror')
        extrapolate = argdef(extrapolate, self.extrapolate, False)
        compute_map = argdef(compute_map, self.compute_map, False)
        ensure_multiple = argdef(ensure_multiple, self.ensure_multiple)
        ensure_multiple = argpad(ensure_multiple, 3)

        # Pad shape if needed
        output_shape = [int(np.ceil(o/m)*m) if m is not None and o != 1 else o
                        for o, m in zip(output_shape, ensure_multiple)]
        # TODO: add little translation so that the padding is central?

        # Change output layout if needed
        if output_layout is not None:
            output_affine, output_shape = change_layout(
                output_affine, output_shape, output_layout)

        # Compute affine map
        Mi = np.asarray(input_affine)
        Mo = np.asarray(output_affine)
        M = lmdiv(Mi, Mo)
        g = affine_grid(M, output_shape, dtype=np.float32)
        input_shape = x.shape[:3]

        # Sample
        if order > 1:
            x = spline_filter1d(x, order=order, axis=0, mode=bound, output=x)
            x = spline_filter1d(x, order=order, axis=1, mode=bound, output=x)
            x = spline_filter1d(x, order=order, axis=2, mode=bound, output=x)
        # x = sample_grid(x, g, order, bound)
        x = map_coordinates(x, g.transpose((3, 0, 1, 2)),
                            order=order, mode=bound, prefilter=False)

        # Mask of out-of-bound voxels
        if not extrapolate:
            gap = 0.5  # TODO: What 's best?
            msk = (g[..., 0] < -gap) | (g[..., 0] > input_shape[0]-1.0+gap) \
                | (g[..., 1] < -gap) | (g[..., 1] > input_shape[1]-1.0+gap) \
                | (g[..., 2] < -gap) | (g[..., 2] > input_shape[2]-1.0+gap)
            x[msk] = 0

        x = writer(x, info=info, affine=output_affine)

        if compute_map:
            map = reliability_grid(g)
            if not extrapolate:
                map[msk] = 0
            map = map_writer(map, info=info, affine=output_affine)
            return x, map
        else:
            return x


class ReslicerLike(Reslicer):
    """Reslice a volume to the space of another volume."""

    def __init__(self, reference_volume=None, **kwargs):
        super().__init__(**kwargs)
        self.reference_volume = reference_volume

    def __call__(self, x, reference_volume=None, input_affine=None, **kwargs):
        """Reslice a volume to the space of another volume.

        Parameters
        ----------
        x : str or array_like
            Input volume.

        reference_volume : fiel_like or array_like
            Reference volume, whose shape and affine matrix define the
            reference space

        input_affine : matrix_like, default=guessed from input
            Input orientation matrix, mapping voxels to world space

        Other Parameters
        ----------------
        order : int, default=self.order
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar,
                default=self.bound
            Boundary conditions when sampling out-of-bounds

        extrapolate : bool, default=self.extrapolate
            Use boundary conditions to extrapolate data outside of
            the original field-of-view.

        compute_map : bool, default=self.compute_map
            Compute reliability map

        ensure_multiple : vector_like[int], optional
            Ensures that the shape is a multiple of a value (e.g., 32).
            This can be useful for convolutional neural networks.
            If used, pad non-singleton dimensions until they fulfill
            the condition.

        writer : io.VolumeWriter, default=self.writer
            Writer object for the resliced image

        map_writer : io.VolumeWriter, default=self.map_writer
            Writer object for the reliability map

        Returns
        -------
        y : np.ndarray
            Resliced volume

        map : file_like or np.ndarray, optional
            Reliability map

        """

        if reference_volume is None:
            reference_volume = self.reference_volume
        if reference_volume is None:
            raise ValueError('No reference volume provided')
        info = self.reader.inspect(reference_volume)
        output_affine = info.get('affine')
        output_shape = info.get('shape')

        # Specify default for extrapolate
        if kwargs.get('extrapolate') is None:
            kwargs['extrapolate'] = self.extrapolate
        if kwargs.get('extrapolate')  is None:
            kwargs['extrapolate'] = False

        return super().__call__(x,
                                output_affine=output_affine,
                                output_shape=output_shape,
                                input_affine=input_affine,
                                **kwargs)


class Resizer(Reslicer):
    """Resize an image by a given factor (greater or lower than one).

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When upsampling by
    an integer factor or downsampling by a divisor of the input size,
    the bottom right corners should also match.

    """

    output_prefix = 'resized_'

    @staticmethod
    def _transform_factor(f):
        return f

    def __init__(self, factor=None, output_shape=None, **kwargs):
        """

        Parameters
        ----------
        factor : iterable, optional
            Factor by which to scale the voxel size

        output_shape : iterable, default=from factor
            Output shape

        Other Parameters
        ----------------
        order : int, default=1
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='nearest'
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=False
            Compute reliability map

        ensure_multiple : vector_like[int], optional
            Ensures that the shape is a multiple of a value (e.g., 32).
            This can be useful for convolutional neural networks.
            If used, pad non-singleton dimensions until they fulfill
            the condition.

        writer : io.VolumeWriter, optional
            Writer object for the resliced image

        map_writer : io.VolumeWriter, optional
            Writer object for the reliability map

        """
        super().__init__(**kwargs)
        self.factor = factor
        self.output_shape = output_shape

    def __call__(self, x, factor=None, output_shape=None, **kwargs):
        """

        Parameters
        ----------
        x : str or array_like
            Input volume.

        factor : iterable, default=self.factor
            Factor by which to scale the voxel size

        output_shape : iterable, default=self.output_shape
            Output shape

        Other Parameters
        ----------------
        order : int, default=self.order
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar,
                default=self.bound
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=self.compute_map
            Compute reliability map

        ensure_multiple : vector_like[int], default=self.ensure_multiple
            Ensures that the shape is a multiple of a value (e.g., 32).
            This can be useful for convolutional neural networks.
            If used, pad non-singleton dimensions until they fulfill
            the condition.

        writer : io.VolumeWriter, default=self.writer
            Writer object for the resliced image

        map_writer : io.VolumeWriter, default=self.map_writer
            Writer object for the reliability map

        Returns
        -------
        y : array_like
            Output volume

        """

        info = self.reader.inspect(x)
        input_shape = info.get('shape')
        input_affine = info.get('affine')

        if factor is None:
            factor = self.factor
        factor = argpad(factor, 3)
        factor = [self._transform_factor(f) for f in factor]

        # Compute output affine
        # We want edges (i.e., top-left corners) to be aligned.
        # > We therefore first apply a negative translation of half a
        #   voxel; then scale; then apply a positive translation of
        #   half a (scaled) voxel.
        scale = np.diag(factor + [1])
        shift = np.eye(4)
        shift[:3, 3] = 0.5
        scale = np.matmul(scale, shift)
        shift[:3, 3] = -0.5
        scale = np.matmul(shift, scale)
        output_affine = np.matmul(input_affine, scale)

        # Compute output shape
        if output_shape is None:
            output_shape = self.output_shape
        if output_shape is None:
            output_shape = [np.floor(i/f) for i, f in zip(input_shape, factor)]

        return super().__call__(x, input_affine=input_affine,
                                output_affine=output_affine,
                                output_shape=output_shape,
                                **kwargs)


class Upsampler(Resizer):
    """Upsample images by a factor.

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When upsampling by
    an integer factor, the bottom right corners should also match.

    """

    output_prefix = 'upsampled_'

    @staticmethod
    def _transform_factor(f):
        return 1/f


class Downsampler(Resizer):
    """Downsample images by a factor.

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When downsampling
    by a divisor of the input size, the bottom right corners should
    also match.

    """

    output_prefix = 'downsampled_'


class ShapeResizer(Resizer):
    """Resize the shape of an image to match a target shape.

    The top-left and bottom right corners of both field-of-views are
    used as anchors.

    """

    def __init__(self, output_shape=None, **kwargs):
        """

        Parameters
        ----------
        output_shape : iterable, optional
            Output shape

        Other Parameters
        ----------------
        order : int, default=1
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='nearest'
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=False
            Compute reliability map

        ensure_multiple : vector_like[int], optional
            Ensures that the shape is a multiple of a value (e.g., 32).
            This can be useful for convolutional neural networks.
            If used, pad non-singleton dimensions until they fulfill
            the condition.

        writer : io.VolumeWriter, optional
            Writer object for the resliced image

        map_writer : io.VolumeWriter, optional
            Writer object for the reliability map

        """
        super().__init__(**kwargs)
        self.output_shape = output_shape

    def __call__(self, x, output_shape=None, **kwargs):
        """

        Parameters
        ----------
        x : str or array_like
            Input volume.

        output_shape : iterable, default=self.output_shape
            Output shape

        Other Parameters
        ----------------
        order : int, default=self.order
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar,
                default=self.bound
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=self.compute_map
            Compute reliability map

        ensure_multiple : vector_like[int], default=self.ensure_multiple
            Ensures that the shape is a multiple of a value (e.g., 32).
            This can be useful for convolutional neural networks.
            If used, pad non-singleton dimensions until they fulfill
            the condition.

        writer : io.VolumeWriter, default=self.writer
            Writer object for the resliced image

        map_writer : io.VolumeWriter, default=self.map_writer
            Writer object for the reliability map

        Returns
        -------
        y : array_like
            Output volume

        """

        info = self.reader.inspect(x)
        input_shape = info.get('shape')

        if output_shape is None:
            output_shape = self.output_shape
        output_shape = argpad(output_shape, 3)

        # Compute factor
        factor = [i/o for o, i in zip(output_shape, input_shape)]

        # Call resizer
        return super().__call__(x, factor, output_shape, **kwargs)


class VoxelResizer(Resizer):
    """Resize the voxel size of an image to match a target voxel size.

    The top-left and bottom right corners of both field-of-views are
    used as anchors.

    """

    output_prefix = 'rescaled_'

    def __init__(self, output_vs=None, **kwargs):
        """

        Parameters
        ----------
        output_vs : iterable, optional
            Output voxel size

        Other Parameters
        ----------------
        order : int, default=1
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default='nearest'
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=False
            Compute reliability map

        ensure_multiple : vector_like[int], optional
            Ensures that the shape is a multiple of a value (e.g., 32).
            This can be useful for convolutional neural networks.
            If used, pad non-singleton dimensions until they fulfill
            the condition.

        writer : io.VolumeWriter, optional
            Writer object for the resliced image

        map_writer : io.VolumeWriter, optional
            Writer object for the reliability map

        """
        super().__init__(**kwargs)
        self.output_vs = output_vs

    def __call__(self, x, output_vs=None, **kwargs):
        """

        Parameters
        ----------
        x : str or array_like
            Input volume.

        output_vs : iterable, default=self.output_shape
            Output voxel size

        Other Parameters
        ----------------
        order : int, default=self.order
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar,
                default=self.bound
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=self.compute_map
            Compute reliability map

        ensure_multiple : vector_like[int], default=self.ensure_multiple
            Ensures that the shape is a multiple of a value (e.g., 32).
            This can be useful for convolutional neural networks.
            If used, pad non-singleton dimensions until they fulfill
            the condition.

        writer : io.VolumeWriter, default=self.writer
            Writer object for the resliced image

        map_writer : io.VolumeWriter, default=self.map_writer
            Writer object for the reliability map

        Returns
        -------
        y : array_like
            Output volume

        """

        info = self.reader.inspect(x)
        input_vs = np.sqrt((info.get('affine')[:3, :3] ** 2).sum(axis=0))

        if output_vs is None:
            output_vs = self.output_vs
        output_vs = argpad(output_vs, 3)

        # Compute factor
        factor = [o/i for o, i in zip(output_vs, input_vs)]

        # Call resizer
        return super().__call__(x, factor, **kwargs)
