from argparse import ArgumentParser
from ..utils import expand_images
from ..io import VolumeWriter
from .object import Reorienter, Inspecter
import numpy as np

#                           --------------
#                           Common options
#                           --------------
# These options are common to all sub commands.
input = ArgumentParser(add_help=False)
input.add_argument('--images', '-i', nargs='+', required=True,
                    metavar='IMAGE', help='Input images')
output = ArgumentParser(add_help=False)
output.add_argument('--output-dtype', '-dt', default=None,
                    dest='output_dtype', metavar='TYPE',
                    help='Output data type [default: same as input]')
output.add_argument('--output-dir', '-o', default=None,
                    dest='output_dir', metavar='DIR',
                    help='Output directory [default: same as input]')
output.add_argument('--output-prefix', '-p', default=None,
                    dest='output_prefix', metavar='PREFIX',
                    help='Output prefix [default: <LAYOUT>_]')
output.add_argument('--output-format', '-f', default=None,
                    dest='output_ext', metavar='FORMAT',
                    help='Output extension [default: same as input]')

#                           ------------
#                           Sub commands
#                           ------------
parser = ArgumentParser(prog='nntools.space')
sub = parser.add_subparsers()
# ---
# reorient
# ---
reo = sub.add_parser('reorient', parents=[input, output],
                     help='Reorient a volume to match a target layout')
reo.add_argument('layout', type=str, help='Output layout')
reo.set_defaults(klass=Reorienter)
# ---
# inspect
# ---
reo = sub.add_parser('inspect', parents=[input],
                     help='Inspect the space of a volume')
reo.set_defaults(klass=Inspecter)

#                           -------------
#                           Parse options
#                           -------------

# Parse
args = parser.parse_args()

# Output options
if args.klass is Reorienter:
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
        output_kwargs['prefix'] = args.layout + '_'
    writer = VolumeWriter(**output_kwargs)
    common_kwargs = {
        'writer': writer,
    }

#                           --------------
#                           Prepare object
#                           --------------

# Initialize appropriate object
if args.klass is Reorienter:
    obj = args.klass(layout=args.layout, **common_kwargs)
elif args.klass is Inspecter:
    obj = args.klass()
else:
    raise NotImplementedError

#                           --------------
#                           Process images
#                           --------------

images = expand_images(args.images)
for i in images:
    obj(i)
