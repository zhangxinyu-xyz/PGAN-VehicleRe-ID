import os
try:
   import cPickle as pickle
except:
   import pickle

def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def save_obj(name, obj):
    mkdirs(os.path.dirname(name))
    with open(name, 'wb') as f:
        # print 'saving to: %s' % name
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        # print 'loading from: %s' % name
        return pickle.load(f)