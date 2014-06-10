#coding: utf-8;
import numpy as np
import os
import sys

class Dataset():
    def __init__(self, filename, label_index=0, isNonvectorial=False, **kwargs):
        if not os.path.exists(filename):
            raise IOError("can not find the file named: %s" % filename)

        self.raw = np.loadtxt(filename, **kwargs)

        if not isNonvectorial:
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

    def normalize(self):
        return self.data / np.max(np.fabs(self.data), axis=0)

if __name__ == '__main__':
    Dataset("data/page-blocks.rplcd", label_index=-1, dtype=np.float)
    Dataset("data/yeast.rplcd", label_index=-1, usecols=range(1,10), dtype=np.float)
    Dataset("data/abalone.rplcd", label_index=-1, usecols=range(1,9), delimiter=',', dtype=np.float)
    Dataset("data/ecoli.rplcd", label_index=-1, usecols=range(1,9), dtype=np.float)
    Dataset("data/transfusion.rplcd", label_index=-1, delimiter=',', skiprows=1, dtype=np.float)
    Dataset("data/haberman.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    Dataset("data/waveform.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    Dataset("data/pima-indians-diabetes.rplcd", label_index=-1, delimiter=',', dtype=np.float)
    Dataset("data/SMSSpamCollection.rplcd", isNonvectorial=True, delimiter='\t', dtype={'names':('0','1'), 'formats':('f8','S512')})
