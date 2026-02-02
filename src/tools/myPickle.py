import bz2
import pickle
import _pickle as cPickle

# Load any compressed pickle file
def decompress(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

 # Pickle a file and then compress it into a file with extension 
def compress(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
