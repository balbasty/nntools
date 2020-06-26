import numpy as np
import os.path
from .io import VolumeReader, VolumeWriter

# TODO:
# . more tests
# . other methods (percentiles...)
# . option to not write on disk (if used as a library)
# . commandline: how to deal with list of list of labels?
# . make the code nicer


class ROINormalizer:
    """Normalize based on an aggregate value in a region-of-interest."""

    def __init__(self, labels=None, metric='median', target=1,
                 writer=VolumeWriter(dtype=np.float32, prefix='normalized_')):
        super().__init__()
        self.image_reader = VolumeReader(dtype=np.float32)
        self.label_reader = VolumeReader()
        self.writer = writer
        self.labels = labels
        self.metric = metric
        self.target = target

    def normalize(self, x, labs):
        x, info = self.image_reader(x)
        lab = self.load_label(labs)
        if self.metric == 'mean':
            reference = np.average(x, weights=lab, dtype=np.float64)
        elif self.metric == 'median':
            reference = np.nanmedian(x[lab>0.5])
        else:
            raise TypeError("Metric must be 'mean' or 'median'. Got {}."
                            .format(self.metric))
        x = x * (self.target/reference)
        x = self.writers
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
            lab, _ = self.label_reader(labs[0])
            for l in labs[1:]:
                lab += self.label_reader(l)[0]
        else:
            lab, _ = self.label_reader(labs[0])
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
    from .utils import expand_images

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
        'dtype': args.output_dtype,
        'dir': args.output_dir,
        'prefix': args.output_prefix,
        'ext': args.output_ext,
    }
    if output_args['dtype'] is not None:
        output_args['dtype'] = np.dtype(output_args['dtype'])
    if output_args['ext'] and \
            not output_args['ext'].startswith('.'):
        output_args['ext'] = '.' + output_args['ext']
    writer = VolumeWriter(dtype=output_args['dtype'], dir=output_args['dir'],
                          prefix=output_args['prefix'], ext=output_args['ext'])

    if method == 'roi':
        obj = ROINormalizer(args.label_list, args.metric, args.target,
                            writer=writer)
        if len(images) != len(labels):
            raise TypeError('There should be as many images as label files.')
        for i, l in zip(images, labels):
            obj.normalize(i, l)
