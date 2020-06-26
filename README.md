# nntools
Tools (preprocessing etc.) for computer vision / neural networks

## Usage

``nntools`` can be used as a python library or as a collection of 
scripts. First, move to the project directory. It is then
recommended to install the package using either setuptools or pip:
```shell script
cd /path/to/nntools
python setupy.py install
# OR
pip install .
```
Note that ``nntools`` depends on ``numpy`` and ``nibabel``.

### Run as scripts
Many modules in ``nntools`` are _runnable_ modules, meaning that they 
can be used on the commandline. To access their help, type:
```shell script
python -m nntools.<module> -h
```

The list of runnable modules is:
- ``reslice``
- ``intensity_normalize``

### Import as modules

In python, just start with
```python
from nntools import <module>
```
