#!/bin/bash
CURRENT_DIR=`pwd`

UTILS_ROOT=`realpath $(dirname $0)`/utils
MODULES_ROOT=$UTILS_ROOT/modules_to_build
MAPPER_ROOT=$MODULES_ROOT/dvs-mapping-module

for MODULE_ROOT in `find $MODULES_ROOT -maxdepth 1 ! -path $MODULES_ROOT -type d`
do
  MODULE_BUILD=`mktemp -d`;
  cd $MODULE_BUILD && \
    cmake -DCMAKE_BUILD_TYPE=Release -Doutput_dir=$UTILS_ROOT $MODULE_ROOT && \
    cmake --build . && \
    cd $CURRENT_DIR && \
    rm -rf $MODULE_BUILD
done
