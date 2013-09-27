#coding: utf-8;
import numpy as np
import os
import sys

class Dataset():
    def __init__(self, filename, label_index=0, **kwargs):
        if not os.path.exists(filename):
            raise IOError("can not find the file named: %s" % filename)

        self.raw = np.loadtxt(filename, **kwargs)

        if label_index >= 0:
            self.label = self.raw[:,label_index]
            left = self.raw[:,:label_index:]
            right = self.raw[:,label_index+1:]
            self.data = np.c_[left, right]
        else:
            # if given label index is negative,
            # forcibly use -1 as index number
            self.label = self.raw[:,-1]
            self.data = self.raw[:,:-1]

if __name__ == '__main__':
    # values in datafile most be preproccessed, using shell cmd like below

    # sed "s/[^5]$/-1/g" page-blocks.data | sed "s/5$/1/g" > page-blocks.rplcd
    Dataset("data/page-blocks.rplcd", label_index=-1, dtype=np.float)

    # sed "s/CYT$/-1/g" yeast.data | sed "s/NUC$/-1/g" | sed "s/MIT$/-1/g" | sed "s/ME3$/-1/g" | sed "s/ME2$/1/g" | sed "s/ME1$/-1/g" | sed "s/EXC$/-1/g" | sed "s/VAC$/-1/g" | sed "s/POX$/-1/g" | sed "s/ERL$/-1/g" > yeast.rplcd
    Dataset("data/yeast.rplcd", label_index=-1, usecols=range(1,10), dtype=np.float)

