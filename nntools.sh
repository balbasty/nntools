#!/bin/bash

# This script is a small utility that allows runnable modules from
# nntools to be called easily.

MODULE_LIST=( "intensity_normalize" "reslice" "space" )

ERROR_MSG_BAD_MODULE="Module name not recognised. Should be one of ( "
ERROR_MSG_BAD_MODULE+="${MODULE_LIST[*]}"
ERROR_MSG_BAD_MODULE+=" )"

ERROR_MSG_NO_MODULE="No module provided. Should be one of ( "
ERROR_MSG_NO_MODULE+="${MODULE_LIST[*]}"
ERROR_MSG_NO_MODULE+=" )"

HELP_MSG="usage: nntools.sh [-h] { "
HELP_MSG+="${MODULE_LIST[*]}"
HELP_MSG+=" }"

MIN_PYTHON_VERSION="350"

# Check python version
PYTHON_VERSION=$(python -V 2>&1 | grep -o '[0-9\.]\+')
PYTHON_VERSION=$(echo "${PYTHON_VERSION//./}")
if [ "${#PYTHON_VERSION}" -gt 3 ]; then
  EXTRA="${#PYTHON_VERSION}"
  EXTRA=$((EXTRA - 3))
  for i in $(seq 1 1 "$EXTRA"); do
    MIN_PYTHON_VERSION+="0"
  done
fi
ERROR_MSG_PYTHON="Invalid python version: should be >= "
ERROR_MSG_PYTHON+="${MIN_PYTHON_VERSION}"
ERROR_MSG_PYTHON+=" but got "
ERROR_MSG_PYTHON+="${PYTHON_VERSION}"
[ "$PYTHON_VERSION" -ge "$MIN_PYTHON_VERSION" ] \
  || { echo "$ERROR_MSG_PYTHON" >&2; exit 1; }

# If no module is provided, return with an error
[ "$#" -eq 0 ] && { echo "$ERROR_MSG_NO_MODULE" >&2; exit 1; }

# Find the script directory and add it to the python path to expose nntools
SCRIPT=$(readlink "$0")
BASEDIR=$(dirname "$SCRIPT")
if [ -z "$PYTHONPATH" ]; then
  export PYTHONPATH="$BASEDIR"
else
  export PYTHONPATH="$BASEDIR:$PYTHONPATH"
fi

# Read module name and check that it is correct
MODULE="$1"
{ [ "$MODULE" = "-h" ] || [ "$MODULE" = "--help" ]; }  && { echo "$HELP_MSG"; exit 0; }
SUCCESS=""
for CORRECT_MODULE in "${MODULE_LIST[@]}"; do
  [ "$MODULE" = "$CORRECT_MODULE" ] && SUCCESS=1 && break
done
[ -z "$SUCCESS" ] && { echo "$ERROR_MSG_BAD_MODULE" >&2; exit 1; }

# Say which nntools we are using
COMMAND="import nntools; import os; "
COMMAND+="print('Using: {}'.format(os.path.dirname(nntools.__file__)))"
python -c "$(echo "$COMMAND")" \
  || { echo "Failed to import nntools" >&2; exit 1; }

# Remove module name from the argument list and run the module
shift  1
python -m nntools."$MODULE" "$@"
