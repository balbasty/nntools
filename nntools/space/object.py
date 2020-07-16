"""Utilities related to voxel/world spaces implemented in an
Object-Oriented paradigm."""

import numpy as np
import os
from ..io import VolumeReader, VolumeWriter, VolumeConverter, isfile
from ..utils import argdef
from .affine import change_layout, voxel_size, affine_layout


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

    def __call__(self, x, affine=None, layout=None, writer=None):
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


class Inspecter:
    """Inspect the space of a volume or file."""

    fields = ('shape', 'voxel size', 'layout', 'affine')
    field_len = max(*(len(field) for field in fields)) + 1

    def __init__(self):
        self.reader = VolumeReader()

    def __call__(self, x):
        """

        Parameters
        ----------
        x : file_like or array_like
            Input volume

        """

        info = self.reader.inspect(x)

        if isinstance(x, np.ndarray):
            print('Array')
        else:
            fname = os.path.join(info.get('dir'),
                                 info.get('basename') + info.get('ext'))
            shape = info.get('shape')
            affine = info.get('affine')
            vs = voxel_size(affine)
            layout = affine_layout(affine)

            print('File: {}'.format(fname))
            self.print_field('shape', shape)
            self.print_field('voxel size', vs)
            self.print_field('layout', layout)
            self.print_field('affine', affine)

    def print_field(self, name, value, format=''):
        repr_value = '{' + format + '}'
        repr_value = repr_value.format(value).split('\n')
        for n_line in range(1, len(repr_value)):
            pad = ' ' * (self.field_len+1)
            repr_value[n_line] = '\t' + pad + repr_value[n_line]
        repr_value = '\n'.join(repr_value)

        print(('\t{:' + str(self.field_len) + 's} {}')
              .format(name + ':', repr_value))
