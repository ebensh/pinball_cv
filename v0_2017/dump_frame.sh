#!/bin/bash

set -ex

ffmpeg -i $1 -ss $2 -vframes 1 $3
