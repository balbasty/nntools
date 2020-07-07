"""Utilities related to voxel/world spaces."""

from ..io import VolumeReader, VolumeWriter, VolumeConverter, isfile
from ..utils import argdef

from .affine import affine_layout_matrix, affine_layout, change_layout, \
    affine_basis, affine_subbasis, affine_matrix, affine_parameters, \
    mean_affine, mean_space, voxel_size

# ----------------------------------------------------------------------
#                               REORIENT
# ----------------------------------------------------------------------


def _select_writer(x, writer, prefix):
    """Choose writer based on input type."""
    if isfile(x):
        if writer is None:
            writer = VolumeWriter(prefix=prefix)
    else:
        x = np.asarray(x)
        if writer is None:
            writer = VolumeConverter(x.dtype)
    return writer


class Reorienter:
    """Reorient a volume to match a target layout."""

    def __init__(self, layout='RAS', writer=None):
        """

        Parameters
        ----------
        layout : str, default='RAS'
            Name of a layout. See ``affine_layout_matrix``.
        writer : VolumeWriter, optional
            Writer object. Selected based on the input type by default.
        """
        self.layout = layout
        self.reader = VolumeReader()
        self.writer = writer

    def __call__(self, x, affine = None, layout=None, writer=None):
        """

        Parameters
        ----------
        x : file_like or array_like
            Input volume
        affine : matrix_like, default=from input
            Input orientation matrix
        layout : str, default=self.layout
            Name of a layout. See ``affine_layout_matrix``.
        writer : VolumeWriter, default=self.writer

        Returns
        -------
        y : type(x)
            Reoriented volume.

        """

        # Output layout
        layout = argdef(layout, self.layout)

        # Select appropriate writer
        writer = _select_writer(x, self.writer, layout + '_')

        # Load input volume
        x, info = self.reader(x)
        affine = argdef(affine, info.get('affine'))
        if affine is None:
            # Cannot reorient without an affine
            return writer(x, info=info)

        # Reorient
        affine, x = change_layout(affine, x, layout)

        # Write
        return writer(x, info=info, affine=affine)


def reorient(x, affine=None, layout=None, **kwargs):
    """Reorient a volume to match a target layout.

    Parameters
    ----------
    layout : str, default='RAS'
        Name of a layout. See ``affine_layout_matrix``.

    affine : matrix_like, default=from input
        Input orientation matrix

    writer : VolumeWriter, optional
        Writer object. Selected based on the input type by default.

    Returns
    -------
    y : type(x)
        Reoriented volume.

    """
    reorienter = Reorienter()
    return reorienter(x, affine=affine, layout=layout, **kwargs)


# ----------------------------------------------------------------------
#                          COMMAND LINE VERSION
# ----------------------------------------------------------------------


if __name__ == '__main__':
    import os.path
    from argparse import ArgumentParser
    from ..utils import expand_images
    from ..io import VolumeWriter
    import numpy as np

    #                           --------------
    #                           Common options
    #                           --------------
    # These options are common to all sub commands.
    common = ArgumentParser(add_help=False)
    common.add_argument('--images', '-i', nargs='+', required=True,
                        metavar='IMAGE', help='Input images')
    common.add_argument('--output-dtype', '-dt', default=None,
                        dest='output_dtype', metavar='TYPE',
                        help='Output data type [default: same as input]')
    common.add_argument('--output-dir', '-o', default=None,
                        dest='output_dir', metavar='DIR',
                        help='Output directory [default: same as input]')
    common.add_argument('--output-prefix', '-p', default=None,
                        dest='output_prefix', metavar='PREFIX',
                        help='Output prefix [default: resliced_]')
    common.add_argument('--output-format', '-f', default=None,
                        dest='output_ext', metavar='FORMAT',
                        help='Output extension [default: same as input]')

    #                           ------------
    #                           Sub commands
    #                           ------------
    parser = ArgumentParser()
    sub = parser.add_subparsers()
    # ---
    # reorient
    # ---
    reo = sub.add_parser('reorient', parents=[common],
                         help='Reorient a volume to match a target layout')
    reo.add_argument('layout',  type=str, help='Output layout')
    reo.set_defaults(klass=Reorienter)

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
        output_kwargs['prefix'] = args.layout + '_'
    writer = VolumeWriter(**output_kwargs)

    #                           --------------
    #                           Prepare object
    #                           --------------

    # Initialize appropriate object
    common_kwargs = {
        'writer': writer,
    }
    if args.klass is Reorienter:
        obj = args.klass(layout=args.layout, **common_kwargs)
    else:
        raise NotImplementedError

    #                           --------------
    #                           Process images
    #                           --------------

    images = expand_images(args.images)
    for i in images:
        obj(i)
