from setuptools import setup, find_packages
from distutils import log
from distutils.core import Command
import subprocess
import os


class BuildDoc(Command):

    description = 'build documentation'
    user_options = [
        ('parse-api', None, 'parse sources to generate API documentation')
    ]

    def initialize_options(self):
        self.parse_api = False

    def finalize_options(self):
        if self.parse_api is None:
            self.parse_api = False

    def run(self):
        curdir = os.path.abspath(os.curdir)
        projdir = os.path.dirname(os.path.abspath(__file__))
        packdir = os.path.join(projdir, 'nntools')
        docdir = os.path.join(projdir, 'docs')
        APIDOC = 'sphinx-apidoc'
        try:
            os.chdir(docdir)
            if self.parse_api:
                cmd = [APIDOC]
                cmd += ['-o', '_sources']   # Output path
                cmd += ['--force']          # Module before submodule
                cmd += ['--module-first']   # Module before submodule
                cmd += ['--separate']       # Each module on its own page
                cmd += [packdir]            # Path to modules
                self.announce('Running command: %s' % str(cmd), level=log.INFO)
                subprocess.check_call(cmd)
            subprocess.check_call(['make', 'html'])
        finally:
            os.chdir(curdir)


setup(
    name='nntools',
    version='0.1a',
    packages=find_packages(),
    url='https://github.com/balbasty/nntools',
    license='MIT',
    author='Yael Balbastre',
    author_email='yael.balbastre@gmail.com',
    description='Preprocessing tools for computer vision / neural networks',
    python_requires='>=3.5',
    install_requires=['nibabel', 'numpy', 'scipy'],
    cmdclass={'docs': BuildDoc}
)
