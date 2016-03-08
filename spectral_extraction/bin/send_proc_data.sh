#! /bin/bash
# Script to copy data from minerva system

datestamp = '20160216' #Make this an argument input?

if [ -z "{$MINERVA_REDUX_DIR+x}" ]; then
    echo "Error, must set 'MINERVA_REDUX_DIR'"
    return
fi

rsync -azh $MINERVA_REDUX_DIR/n$datestamp  -e "ssh -p 22222" minerva@minerva.sao.arizona.edu:/Data/kiwispec-proc/

