#!/bin/sh

# Change this to the name of pip program for Python 3
PIP3=pip3
# Change this to the name of Python 3 program
PYTHON3=python3

$PIP3 install --user tensorflow imageio
python3 -c 'import imageio; imageio.plugins.ffmpeg.download()'
