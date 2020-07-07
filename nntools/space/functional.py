"""Utilities related to voxel/world spaces implemented in a
Functional paradigm."""

from .object import Reorienter


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
