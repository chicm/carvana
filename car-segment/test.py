
import bcolz
import numpy as np

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]


probs = np.load('/home/chicm/ml/kgdata/carvana/results/single/UNet_double_1024_5/submit/049/probs-part10.8.npy')
save_array('/home/chicm/ml/kgdata/carvana/results/single/UNet_double_1024_5/submit/ensemble/probs-part10.8.npy', probs)