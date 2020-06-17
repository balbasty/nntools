import numpy as np
import nibabel as nb
import os.path

# TODO:
# . more tests
# . other methods (percentiles...)
# . option to not write on disk (if used as a library)
# . commandline: how to deal with list of list of labels?
# . make the code nicer


class Normalizer:
    """Base class for normalizers."""

    def __init__(self, output_ext=None, output_dtype=None, output_dir=None,
                 output_prefix=None):
        self.output_ext = output_ext
        self.output_dtype = output_dtype
        self.output_dir = output_dir
        self.output_prefix = output_prefix

    @staticmethod
    def load(x):
        info = {
            'fname': None,
            'dir': None,
            'ext': None,
            'dtype': None,
            'class': None,
        }
        if isinstance(x, str):
            fname = x
            info['dir'] = os.path.dirname(fname)
            fname = os.path.basename(fname)
            fname, ext = os.path.splitext(fname)
            if ext == '.gz':
                fname, ext0 = os.path.splitext(fname)
                ext = ext0 + ext
            info['fname'] = fname
            info['ext'] = ext
            x = nb.load(x)
        if isinstance(x, nb.spatialimages.SpatialImage):
            info['dtype'] = x.get_data_dtype()
            info['class'] = x
            x = x.get_fdata(dtype=np.float32)
        if not isinstance(x, np.ndarray):
            raise TypeError("Input type '{}' not handled".format(type(x)))
        return x, info

    def write(self, x, info):
        output_ext = self.output_ext if self.output_ext \
                     else info['ext'] if info['ext'] \
                     else 'nii.gz'
        output_dir = self.output_dir if self.output_dir \
                     else info['dir'] if info['dir']  \
                     else '.'
        output_dtype = self.output_dtype if self.output_dtype  \
                       else info['dtype'] if info['dtype']  \
                       else x.dtype
        output_fname = (self.output_prefix if self.output_prefix
                        else '') + (info['fname'] if info['fname'] else '')
        affine = info['class'].affine if info['class'] else None
        header = info['class'].header if info['class'] else None
        extra = info['class'].extra if info['class'] else None

        if len(x.shape) > 3 and x.shape[3] == 1:
            x = x.reshape(x.shape[:3])
        nbobj = nb.spatialimages.SpatialImage(x, affine, header, extra)
        nbobj.header.set_data_dtype(output_dtype)
        nbobj.header.set_data_shape(x.shape)

        output_path = os.path.join(output_dir, output_fname + output_ext)
        nb.save(nbobj, output_path)
        return nb.load(output_path)

    def normalize(self, x, *args, **kwargs):
        x, info = self.load(x)
        x = self._normalize(x, *args, **kwargs)
        x = self.write(x, info)
        return x

    def _normalize(self, *args, **kwargs):
        raise NotImplementedError('Abstract method _normalize not implemented')


class ROINormalizer(Normalizer):
    """Normalize based on an aggregate value in a region-of-interest."""

    def __init__(self, labels=None, metric='median', target=1,
                 output_ext=None, output_dtype=None, output_dir=None,
                 output_prefix=None):
        super().__init__(output_ext, output_dtype, output_dir, output_prefix)
        self.labels = labels
        self.metric = metric
        self.target = target

    def _normalize(self, x, labs):
        lab = self.load_label(labs)
        if self.metric == 'mean':
            reference = np.average(x, weights=lab, dtype=np.float64)
        elif self.metric == 'median':
            reference = np.nanmedian(x[lab>0.5])
        else:
            raise TypeError("Metric must be 'mean' or 'median'. Got {}."
                            .format(self.metric))
        x = x * (self.target/reference)
        return x

    def load_label(self, labs):
        if isinstance(labs, tuple):
            labs = list(labs)
        elif not isinstance(labs, list):
            labs = [labs]
        if len(labs) > 1:
            # Assume list of responsibilities
            if self.labels is not None:
                labs = [labs[i] for i in self.labels]
            lab, _ = self.load(labs[0])
            for l in labs[1:]:
                lab += self.load(l)[0]
        else:
            lab, _ = self.load(labs[0])
            if len(lab.shape) > 3 and lab.shape[3] > 1:
                # Assume 4D volume of responsibilities
                if self.labels is not None:
                    lab = lab[:, :, :, self.labels]
                lab = lab.sum(axis=3)
            else:
                if self.labels is not None:
                    # Assume hard labels
                    lab = lab.astype(np.int)
                    lab = np.isin(lab, self.labels)

        lab = lab.reshape(lab.shape[:3])
        return lab

# ----------------------------------------------------------------------
#                          COMMAND LINE VERSION
# ----------------------------------------------------------------------


if __name__ == '__main__':
    from argparse import ArgumentParser
    from glob import glob

    # This bit gives some flexibility for the input files:
    # . unix-style ~ (HOME) is always expanded
    # . unix-style tokens are always expanded
    # . if an input file is a directories, we look for all image files
    #   within. A list of supported format is specified below. In
    #  theory, we could use all extensions supported by nibabel.
    possible_extensions = ['.nii', '.nii.gz', '.mgh', '.mgz']
    expansion_pattern = '*['
    for ext in possible_extensions[:-1]:
        expansion_pattern += ext + '|'
    expansion_pattern += ext[-1] + ']'

    def expand_images(images):
        oimages = []
        for entry in images:
            # entry is: image or dir or token-images or token-dirs
            entry = sorted(glob(os.path.expanduser(entry)))
            for subentry in entry:
                if os.path.isdir(subentry):
                    # expand + sort
                    subentry = glob(os.path.join(subentry, expansion_pattern)).sort()
                    oimages += subentry
                else:
                    oimages += [subentry]
        return oimages

    parser = ArgumentParser()

    # Positional arguments
    parser.add_argument('--images', '-i', nargs='+', required=True)
    parser.add_argument('--method', '-m', choices=['roi'], default='roi')
    parser.add_argument('--labels', '-l', nargs='*')
    parser.add_argument('--label-list', type=int, nargs='+', dest='label_list')
    parser.add_argument('--metric', choices=['mean', 'median'], default='median')
    parser.add_argument('--target', type=float, default=1.0)
    parser.add_argument('--output-dtype', '-dt', default=None, dest='output_dtype')
    parser.add_argument('--output-dir', '-o', default=None, dest='output_dir')
    parser.add_argument('--output-prefix', '-p', default=None, dest='output_prefix')
    parser.add_argument('--output-format', '-f', default=None, dest='output_ext')

    args = parser.parse_args()
    images = expand_images(args.images)
    labels = expand_images(args.labels)
    method = args.method
    output_args = {
        'output_dtype': args.output_dtype,
        'output_dir': args.output_dir,
        'output_prefix': args.output_prefix,
        'output_ext': args.output_ext,
    }
    if output_args['output_dtype'] is not None:
        output_args['output_dtype'] = np.dtype(output_args['output_dtype'])
    if output_args['output_ext'] and \
            not output_args['output_ext'].startswith('.'):
        output_args['output_ext'] = '.' + output_args['output_ext']

    if method == 'roi':
        obj = ROINormalizer(args.label_list, args.metric, args.target,
                            **output_args)
        if len(images) != len(labels):
            raise TypeError('There should be as many images as label files.')
        for i, l in zip(images, labels):
            obj.normalize(i, l)



