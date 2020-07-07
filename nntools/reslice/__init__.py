"""Tools for reslicing volumes.

Reslicing is the sequential process of:
    * **interpolation:**  transform a discrete set of points into a
      continuous function;
    * **spatial transformation:** compose the continuous image
      function with a spatial transform *i.e.*, a change of coordinates;
    * **resampling:** evaluate the transformed function at a new
      set of discrete points.

"""

from .object import Reslicer, ReslicerLike, Resizer, ShapeResizer, \
                    VoxelResizer, Upsampler, Downsampler
from .functional import reslice, reslice_like, resize, resize_shape, \
                        resize_voxel, upsample, downsample
