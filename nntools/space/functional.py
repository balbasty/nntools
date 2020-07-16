"""Utilities related to voxel/world spaces implemented in a
Functional paradigm."""

from .object import Reorienter, Inspecter


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
    return Reorienter()(x, affine=affine, layout=layout, **kwargs)


def inspect(x):
    """Inspect the space of a volume.

    Parameters
    ----------
        x : file_like or array_like
            Input volume

    """
    return Inspecter()(x)
