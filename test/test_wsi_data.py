
import sys
import os
sys.path.append(os.getcwd())

# from dataset.wsi_lists import camlon16_data
from dataset.wsi_reader import camlon16_wsis
# from conf.camlon16 import  camlon16_label_fn
from conf.camlon16 import settings



def test_camlon16():
    # for wis_dir in train_dirs['wsis']:

        # for wsi in camlon16_data(wsi_dir=wis_dir):

    # for wsi in camlon16_data(wsi_filenames=camlon16_wsi_filenames(data_set='train'), label_fn=camlon16_label_fn):
    for wsi in camlon16_wsis('train'):
        print(wsi)




    import time
    time.sleep(10)


test_camlon16()
