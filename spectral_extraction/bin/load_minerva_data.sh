#! /bin/bash
# Script to copy data from minerva system
array=($(ssh -p 22222 minerva@minerva.sao.arizona.edu "ls /Data/kiwispec/n201603*/*.fits"))

if [ -z "{$MINERVA_DATA_DIR+x}" ]; then
    echo "Error, must set 'MINERVA_DATA_DIR'"
    return
fi

for i in ${array[@]}; do
    file=`basename $i`
    base=`dirname $i`
    direc=`basename $base`
    if [ ! -d "$MINERVA_DATA_DIR/$direc" ]; then
        mkdir $MINERVA_DATA_DIR/$direc
    fi
    if [ ! -f "$MINERVA_DATA_DIR/$direc/$file" ]; then
        rsync -azh -e "ssh -p 22222" minerva@minerva.sao.arizona.edu:$i $MINERVA_DATA_DIR/$direc/$file
        echo "$direc/$file copied to system"
    fi
done
