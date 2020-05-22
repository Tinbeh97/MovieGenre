import pickle


def save_pickle(outpath, obj):
    with open(outpath, 'wb') as fp:
        pickle.dump(obj, fp, protocol=2)


def load_pickle(outpath):
    with open(outpath, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p
    """
    with open(outpath, 'rb') as fp:
        return pickle.load(fp)
    """


def load_json(path):
    import json
    from pprint import pprint
    with open(path) as data_file:    
        data = json.load(data_file)
    return data


def save_json(path, obj):
    import json
    with open(path, 'w') as outfile:
        json.dump(obj, outfile, indent=4)
 

def create_path(path):
    import os
    try:
        os.makedirs(path)
        return True
    except OSError as oe: 
        print('error')
        return False
