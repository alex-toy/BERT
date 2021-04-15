import pickle
import os
import app.config as cf



def to_pickle(item_name, item) :

    file_name = os.path.join(cf.OUTPUTS_MODELS_DIR, item_name)
    outfile = open(file_name,'wb')
    pickle.dump(item,outfile)
    outfile.close()



def from_pickle(item_name) :
    infile = open(os.path.join(cf.OUTPUTS_MODELS_DIR, item_name),'rb')
    item = pickle.load(infile)
    infile.close()
    return item