from setuptools import setup, find_packages

setup(
    name='nntools',
    version='0.1a',
    packages=find_packages(),
    url='https://github.com/balbasty/nntools',
    license='MIT',
    author='Yael Balbastre',
    author_email='yael.balbastre@gmail.com',
    description='Preprocessing tools for computer vision / neural networks',
    python_requires='>=3',
    install_requires=['nibabel', 'numpy']
)
