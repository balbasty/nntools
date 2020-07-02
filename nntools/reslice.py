"""Tools for reslicing volumes.

Reslicing is the sequential process of:
- interpolation          -> transform a discrete set of points into a
                            continuous function;
- spatial transformation -> compose the continous image function with
                            a spatial transform _i.e._, a change of
                            coordinates;
- resampling             -> evaluate the transformed function at a new
                            set of discrete points.

"""


from .io import VolumeReader, VolumeWriter
from .interpolate import affine_grid, sample_grid, reliability_grid
from .utils import argpad
import numpy as np
from scipy.ndimage.interpolation import spline_filter1d
from scipy.ndimage import map_coordinates

# TODO:
# - Possibility to read/write features one at a time to save memory.


class Reslicer:
    """Generic class for reslicing a volume.

    This class uses the input and output affine orientation matrix
    to compute a voxel-to-voxel mapping between two spaces.

    It is use as a base class by most high-level reslicers.
    """

    def __init__(self, output_affine=None, output_shape=None, *,
                 order=1, bound='mirror', extrapolate=False, compute_map=False,
                 writer=VolumeWriter(prefix='resliced_'),
                 map_writer=VolumeWriter(prefix='map_')):
        """

        Parameters
        ----------
        output_affine : matrix_like, default=same as input
            Output orientation matrix, mapping voxels to world space

        output_shape : iterable, default=same as input
            Output (3D spatial) shape

        order : int, default=1
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default=0
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
        """
        self.output_affine = output_affine
        self.output_shape = output_shape
        self.order = order
        self.bound = bound
        self.extrapolate = extrapolate
        self.compute_map = compute_map
        self.reader = VolumeReader()
        self.writer = writer
        self.map_writer = map_writer

    def reslice(self, x,
                input_affine=None, output_affine=None, output_shape=None, *,
                order=None, bound=None, extrapolate=None, compute_map=None):
        """

        Parameters
        ----------
        x : str or array_like
            Input volume.

        input_affine : matrix_like, default=guessed from input
            Input orientation matrix, mapping voxels to world space

        output_affine : matrix_like, default=self.output_affine
            Output orientation matrix, mapping voxels to world space

        output_shape : iterable, default=self.output_shape
            Output (3D spatial) shape

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

        Returns
        -------
        y : array_like
            Output volume

        """

        # Load input volume
        x, info = self.reader(x, dtype=np.float32)

        # Parse options
        if input_affine is None:
            input_affine = info.get('affine')
        if input_affine is None:
            input_affine = np.eye(4)
        if output_affine is None:
            output_affine = self.output_affine
        if output_affine is None:
            output_affine = np.eye(4)
        if output_shape is None:
            output_shape = self.output_shape
        if output_shape is None:
            output_shape = x.shape
        output_shape = output_shape[:3]
        if order is None:
            order = self.order
        if bound is None:
            bound = self.bound
        if extrapolate is None:
            extrapolate = self.extrapolate
        if compute_map is None:
            compute_map = self.compute_map

        # Compute affine map
        Mi = np.array(input_affine)
        Mo = np.array(output_affine)
        M = np.linalg.lstsq(Mi, Mo, rcond=None)[0]
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
            msk = (g[..., 0] < 0) | (g[..., 0] > input_shape[0]-1) \
                | (g[..., 1] < 0) | (g[..., 1] > input_shape[1]-1) \
                | (g[..., 2] < 0) | (g[..., 2] > input_shape[2]-1)
            x[msk] = 0

        x = self.writer(x, info=info, affine=output_affine)

        if compute_map:
            map = reliability_grid(g)
            if not extrapolate:
                map[msk] = 0
            map = self.map_writer(map, info=info, affine=output_affine)
            return x, map
        else:
            return x


class ReslicerLike(Reslicer):
    """Reslice a volue to the space of another volume."""

    def __init__(self, reference_volume=None, **kwargs):
        super().__init__(**kwargs)
        self.reference_volume = reference_volume

    def reslice(self, x, input_affine=None, reference_volume=None, **kwargs):

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

        return super().reslice(x, input_affine=input_affine,
                               output_affine=output_affine,
                               output_shape=output_shape,
                               **kwargs)


class Resizer(Reslicer):
    """Resize an image by a given factor (greater or lower than one).

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When upsampling by
    an integer factor or downsampling by a divisor of the input size,
    the bottom right corners should also match.

    """

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

        order : int, default=1
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default=0
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=False
            Compute reliability map

        writer : io.VolumeWriter, optional
            Writer object for the resliced image

        map_writer : io.VolumeWriter, optional
            Writer object for the reliability map
        """
        super().__init__(**kwargs)
        self.factor = factor
        self.output_shape = output_shape

    def reslice(self, x, factor=None, output_shape=None, **kwargs):
        """

        Parameters
        ----------
        x : str or array_like
            Input volume.

        factor : iterable, default=self.factor
            Factor by which to scale the voxel size

        output_shape : iterable, default=self.output_shape
            Output shape

        order : int, default=self.order
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar,
                default=self.bound
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=self.compute_map
            Compute reliability map

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
        ifactor = [1/f for f in factor]
        scale = np.diag(ifactor + [1])
        shift = np.eye(4)
        shift[:3, 3] = 0.5
        scale = np.dot(scale, shift)
        shift[:3, 3] = -0.5
        scale = np.dot(shift, scale)
        output_affine = np.dot(input_affine, scale)

        # Compute output shape
        if output_shape is None:
            output_shape = self.output_shape
        if output_shape is None:
            output_shape = [np.floor(i*s) for i, s in zip(input_shape, factor)]

        # Specify default for extrapolate
        if kwargs.get('extrapolate') is None:
            kwargs['extrapolate'] = self.extrapolate
        if kwargs.get('extrapolate')  is None:
            kwargs['extrapolate'] = True

        return super().reslice(x, input_affine=input_affine,
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
    pass


class Downsampler(Resizer):
    """Downsample images by a factor.

    The factor relates to voxel sizes; that is, the ratio between input
    and output voxel sizes is defined by the factor. The top-left corner
    of both field-of-views is used as an anchor. When downsampling
    by a divisor of the input size, the bottom right corners should
    also match.

    """
    @staticmethod
    def _transform_factor(f):
        return 1/f


class FOVResizer(Resizer):
    """Resize the FOV of an image to match a target FOV.

    The top-left and bottom right corners of both field-of-views are
    used as anchors.

    """

    def __init__(self, output_shape=None, **kwargs):
        """

        Parameters
        ----------
        output_shape : iterable, optional
            Output shape

        order : int, default=1
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default=0
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=False
            Compute reliability map

        writer : io.VolumeWriter, optional
            Writer object for the resliced image

        map_writer : io.VolumeWriter, optional
            Writer object for the reliability map
        """
        super().__init__(**kwargs)
        self.output_shape = output_shape

    def reslice(self, x, output_shape=None, **kwargs):
        """

        Parameters
        ----------
        x : str or array_like
            Input volume.

        output_shape : iterable, default=self.output_shape
            Output shape

        order : int, default=self.order
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar,
                default=self.bound
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=self.compute_map
            Compute reliability map

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
        factor = [o/i for o, i in zip(output_shape, input_shape)]

        # Call resizer
        return super().reslice(x, factor, output_shape, **kwargs)


class VXResizer(Resizer):
    """Resize the voxel size of an image to match a target voxel size.

    The top-left and bottom right corners of both field-of-views are
    used as anchors.

    """

    def __init__(self, output_vs=None, **kwargs):
        """

        Parameters
        ----------
        output_vs : iterable, optional
            Output voxel size

        order : int, default=1
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar, default=0
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=False
            Compute reliability map

        writer : io.VolumeWriter, optional
            Writer object for the resliced image

        map_writer : io.VolumeWriter, optional
            Writer object for the reliability map
        """
        super().__init__(**kwargs)
        self.output_vs = output_vs

    def reslice(self, x, output_vs=None, **kwargs):
        """

        Parameters
        ----------
        x : str or array_like
            Input volume.

        output_vs : iterable, default=self.output_shape
            Output voxel size

        order : int, default=self.order
            Interpolation order

        bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar,
                default=self.bound
            Boundary conditions when sampling out-of-bounds

        compute_map : bool, default=self.compute_map
            Compute reliability map

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
        factor = [i/o for o, i in zip(output_vs, input_vs)]
        print(input_vs)
        print(output_vs)
        print(factor)

        # Call resizer
        return super().reslice(x, factor, **kwargs)

# ----------------------------------------------------------------------
#                          COMMAND LINE VERSION
# ----------------------------------------------------------------------


if __name__ == '__main__':
    import os.path
    from argparse import ArgumentParser
    from .utils import expand_images

    # Common options
    common = ArgumentParser(add_help=False)
    common.add_argument('--images', '-i', nargs='+', metavar='IMAGE', required=True, help='Input images')
    common.add_argument('--order', type=int, default=1, help='Interpolation order [default: 1]')
    common.add_argument('--bound', choices=['warp', 'nearest', 'mirror', 'reflect', 'zero'], default='mirror', help='Boundary condition [default: mirror]')
    common.add_argument('--extrapolate', type=bool, default=None, help='Extrapolate out-of-bound [default: False if `reference` else True]')
    common.add_argument('--compute-map', default=False, action='store_true', dest='compute_map', help='Compute reliability maps [default: False]')
    common.add_argument('--output-dtype', '-dt', metavar='TYPE', default=None, dest='output_dtype', help='Output data type [default: same as input]')
    common.add_argument('--output-dir', '-o', metavar='DIR', default=None, dest='output_dir', help='Output directory [default: same as input]')
    common.add_argument('--output-prefix', '-p', metavar='PREFIX', default=None, dest='output_prefix', help='Output prefix [default: resliced_]')
    common.add_argument('--map-prefix',  metavar='PREFIX', default=None, dest='map_prefix', help='Output prefix for confidence maps [default: map_]')
    common.add_argument('--output-format', '-f', metavar='FORMAT', default=None, dest='output_ext', help='Output extension [default: same as input]')

    # Methods
    parser = ArgumentParser()
    sub = parser.add_subparsers()
    ref = sub.add_parser('reference', parents=[common], help='Reslice to reference space')
    ref.add_argument('reference', metavar='REF', help='Reference volume')
    ref.set_defaults(klass=ReslicerLike)
    up = sub.add_parser('upsample', parents=[common], help='Upsample volume by a factor')
    up.add_argument('factor', metavar='FACTOR', type=float, nargs='+', help='Upsampling factor')
    up.set_defaults(klass=Upsampler)
    down = sub.add_parser('downsample', parents=[common], help='Downsample volume by a factor')
    down.add_argument('factor', metavar='FACTOR', type=float, nargs='+', help='Upsampling factor')
    down.set_defaults(klass=Downsampler)
    resize = sub.add_parser('resize', parents=[common], help='Resize a volume to match a target shape')
    resize.add_argument('shape', metavar='SHAPE', type=int, nargs='+', help='Output shape')
    resize.set_defaults(klass=FOVResizer)
    resize = sub.add_parser('rescale', parents=[common], help='Rescale a volume to match a target voxel size')
    resize.add_argument('vs', metavar='VOXELSIZE', type=int, nargs='+', help='Output voxel size')
    resize.set_defaults(klass=VXResizer)

    # Parse
    args = parser.parse_args()

    # Output options
    output_kwargs = {
        'dtype': args.output_dtype,
        'dir': args.output_dir,
        'prefix': args.output_prefix,
        'ext': args.output_ext,
    }
    if output_kwargs['dtype'] is not None:
        output_kwargs['dtype'] = np.dtype(output_kwargs['dtype'])
    if output_kwargs['ext'] and not output_kwargs['ext'].startswith('.'):
        output_kwargs['ext'] = '.' + output_kwargs['ext']
    if output_kwargs['prefix'] is None:
        output_kwargs['prefix'] = 'resliced_'
    writer = VolumeWriter(**output_kwargs)

    output_kwargs['prefix'] = args.map_prefix
    if output_kwargs['prefix'] is None:
        output_kwargs['prefix'] = 'map_'
    map_writer = VolumeWriter(**output_kwargs)

    # Initialize appropriate reslicer
    common_kwargs = {
        'order': args.order,
        'bound': args.bound if args.bound != 'zero' else 0,
        'extrapolate': args.extrapolate,
        'compute_map': args.compute_map,
        'writer': writer,
        'map_writer': map_writer,
    }
    if args.klass is ReslicerLike:
        reference = os.path.expanduser(args.reference)
        obj = args.klass(reference_volume=reference, **common_kwargs)
    elif args.klass is Upsampler or args.klass is Downsampler:
        obj = args.klass(factor=args.factor, **common_kwargs)
    elif args.klass is FOVResizer:
        obj = args.klass(output_shape=args.shape, **common_kwargs)
    elif args.klass is VXResizer:
        obj = args.klass(output_vs=args.vs, **common_kwargs)
    else:
        raise NotImplementedError

    # DoIt
    images = expand_images(args.images)
    for i in images:
        obj.reslice(i)
