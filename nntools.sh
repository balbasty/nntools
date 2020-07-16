#!/bin/bash

SCRIPT=$(readlink -f "$0")
BASEDIR=$(dirname "$0")

import os
import sys

projdir = os.path.abs(os.path.dir(__file__))
sys.path.insert(0, projdir)

if len(sys.argv) < 2:
    raise ValueError('No module ')
module = sys.argv[1]
