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

# Remove module name from the argument list and run the module
shift  1
python -m nntools."$MODULE" "$@"
