#!/bin/sh
# DEPENDINGS: GNU-sed

PATH_TO_DIR="./data"

sed 's/[^5]$/-1/g' $PATH_TO_DIR/page-blocks.data | sed "s/5$/1/g" > $PATH_TO_DIR/page-blocks.rplcd

sed 's/CYT$/-1/g' $PATH_TO_DIR/yeast.data | sed "s/NUC$/-1/g" | sed "s/MIT$/-1/g" | sed "s/ME3$/-1/g" | sed "s/ME2$/1/g" | sed "s/ME1$/-1/g" | sed "s/EXC$/-1/g" | sed "s/VAC$/-1/g" | sed "s/POX$/-1/g" | sed "s/ERL$/-1/g" > $PATH_TO_DIR/yeast.rplcd

sed 's/,15$/,p/g' $PATH_TO_DIR/abalone.data | sed 's/,[12]\?[0-9]/,-1/g' | sed 's/,p$/,1/g' > $PATH_TO_DIR/abalone.rplcd

sed 's/im$/1/g' $PATH_TO_DIR/ecoli.data | sed 's/[^1]\{1,3\}$/-1/g' > $PATH_TO_DIR/ecoli.rplcd

sed 's/,1.$/,p/g' $PATH_TO_DIR/transfusion.data| sed 's/,0.$/,-1/g' | sed 's/,p$/,1/g' | sed 's/ //g' > $PATH_TO_DIR/transfusion.rplcd

sed 's/1$/-1/g' $PATH_TO_DIR/haberman.data | sed 's/2$/1/g' > $PATH_TO_DIR/haberman.rplcd

sed 's/,[12]$/,-1/g' $PATH_TO_DIR/waveform.data | sed 's/,0$/,1/g' > $PATH_TO_DIR/waveform.rplcd

sed 's/,0$/,-1/g' $PATH_TO_DIR/pima-indians-diabetes.data > $PATH_TO_DIR/pima-indians-diabetes.rplcd

sed 's/^ham/-1/g' $PATH_TO_DIR/SMSSpamCollection | sed 's/^spam/1/g' > $PATH_TO_DIR/SMSSpamCollection.rplcd
