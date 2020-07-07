from typing import Union, Iterable
import numpy as np
import nibabel as nib

Array = Union[np.ndarray, Iterable, int, float]
Matrix = Array
Vector = Matrix
FileArray = Union[str, nib.spatialimages.SpatialImage]
AnyArray = Union[Array, FileArray]
_Bound = Union[str, int, float]
Bound = Union[_Bound, Iterable[_Bound]]
