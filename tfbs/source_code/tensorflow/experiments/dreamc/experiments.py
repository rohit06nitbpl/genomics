
import sys
sys.path.append('../../')
import data_processing as ddp
import models.tfbs_first_model as mod

def test():
    data_dict = ddp.sample_data_dict()
    model  = mod.TFBSFirstModel()
    model.train(data_dict,batch_size=6,n_epoch=200,max_step=-1,feeder_type='queue')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        test()
    else:
        print 'Error! Many arguements!'



