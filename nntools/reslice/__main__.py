import os.path
import numpy as np
from argparse import ArgumentParser
from ..utils import expand_images
from ..io import VolumeWriter
from .object import ReslicerLike, ShapeResizer, VoxelResizer, \
    Upsampler, Downsampler

#                           --------------
#                           Common options
#                           --------------
# These options are common to all sub commands.
common = ArgumentParser(add_help=False)
common.add_argument('--images', '-i', nargs='+', required=True,
                    metavar='IMAGE', help='Input images')
common.add_argument('--order', type=int, default=1,
                    help='Interpolation order [default: 1]')
common.add_argument('--bound',
                    choices=['warp', 'nearest', 'mirror', 'reflect', 'zero'],
                    default='mirror',
                    help='Boundary condition [default: mirror]')
common.add_argument('--extrapolate', type=bool, default=None,
                    help='Extrapolate out-of-bound '
                         '[default: False]')
common.add_argument('--compute-map', default=False, action='store_true',
                    dest='compute_map',
                    help='Compute reliability maps [default: False]')
common.add_argument('--ensure-multiple', nargs='+', type=int, default=None,
                    dest='ensure_multiple', metavar='MULTIPLE',
                    help='Ensure output shape is a multiple of MULTIPLE')
common.add_argument('--output-dtype', '-dt', default=None,
                    dest='output_dtype', metavar='TYPE',
                    help='Output data type [default: same as input]')
common.add_argument('--output-dir', '-o', default=None,
                    dest='output_dir', metavar='DIR',
                    help='Output directory [default: same as input]')
common.add_argument('--output-prefix', '-p', default=None,
                    dest='output_prefix', metavar='PREFIX',
                    help='Output prefix [default: resliced_]')
common.add_argument('--map-prefix', default=None,
                    dest='map_prefix', metavar='PREFIX',
                    help='Output prefix for confidence maps '
                         '[default: map_]')
common.add_argument('--output-format', '-f', default=None,
                    dest='output_ext', metavar='FORMAT',
                    help='Output extension [default: same as input]')
common.add_argument('--output-layout', default=None,
                    dest='output_layout', metavar='LAYOUT',
                    help='Force output layout (e.g. RAS)')

#                           ------------
#                           Sub commands
#                           ------------
parser = ArgumentParser(prog='nntools.reslice')
sub = parser.add_subparsers()
# ---
# reference = reslice_like
# ---
ref = sub.add_parser('reference', parents=[common],
                     help='Reslice to reference space')
ref.add_argument('reference', metavar='REF', help='Reference volume')
ref.set_defaults(klass=ReslicerLike)
# ---
# upsample
# ---
up = sub.add_parser('upsample', parents=[common],
                    help='Upsample volume by a factor')
up.add_argument('factor', type=float, nargs='+', metavar='FACTOR',
                help='Upsampling factor')
up.set_defaults(klass=Upsampler)
# ---
# downsample
# ---
down = sub.add_parser('downsample', parents=[common],
                      help='Downsample volume by a factor')
down.add_argument('factor', type=float, nargs='+', metavar='FACTOR',
                  help='Upsampling factor')
down.set_defaults(klass=Downsampler)
# ---
# resize_fov
# ---
resize = sub.add_parser('resize_shape', parents=[common],
                        help='Resize a volume to match a target shape')
resize.add_argument('shape', type=int, nargs='+', metavar='SHAPE',
                    help='Output shape')
resize.set_defaults(klass=ShapeResizer)
# ---
# resize_voxel
# ---
rescale = sub.add_parser('resize_voxel', parents=[common],
                         help='Rescale a volume to match a target '
                              'voxel size')
rescale.add_argument('vs', type=int, nargs='+', metavar='VOXELSIZE',
                     help='Output voxel size')
rescale.set_defaults(klass=VoxelResizer)

#                           -------------
#                           Parse options
#                           -------------

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
    output_kwargs['prefix'] = args.klass.output_prefix
writer = VolumeWriter(**output_kwargs)

output_kwargs['prefix'] = args.map_prefix
if output_kwargs['prefix'] is None:
    output_kwargs['prefix'] = 'map_'
map_writer = VolumeWriter(**output_kwargs)

#                           --------------
#                           Prepare object
#                           --------------

# Initialize appropriate reslicer
common_kwargs = {
    'order': args.order,
    'bound': args.bound if args.bound != 'zero' else 0,
    'extrapolate': args.extrapolate,
    'compute_map': args.compute_map,
    'ensure_multiple': args.ensure_multiple,
    'output_layout': args.output_layout,
    'writer': writer,
    'map_writer': map_writer,
}
if args.klass is ReslicerLike:
    reference = os.path.expanduser(args.reference)
    obj = args.klass(reference_volume=reference, **common_kwargs)
elif args.klass is Upsampler or args.klass is Downsampler:
    obj = args.klass(factor=args.factor, **common_kwargs)
elif args.klass is ShapeResizer:
    obj = args.klass(output_shape=args.shape, **common_kwargs)
elif args.klass is VoxelResizer:
    obj = args.klass(output_vs=args.vs, **common_kwargs)
else:
    raise NotImplementedError

#                           --------------
#                           Process images
#                           --------------

images = expand_images(args.images)
for i in images:
    obj(i)
