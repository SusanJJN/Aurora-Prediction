import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys 
sys.path.append("..")
import keras.backend as K
import argparse
from lib.config import *
from lib.dataset import *
from lib.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    
    parser.add_argument('--input', 
                        help='input frames',
                        required=True,
                        type=int)
    parser.add_argument('--model_no',
                        help='model index',
                        required=True,
                        type=list)
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    basic_frames = 10
    interval_frames = args.input
    
    basic_images, next_images = read_data(basic_frames, interval_frames, TEST_NPZ_PATH)
    val_basic = basic_images[:,:args.input]
    val_next = next_images

    for k in range(10):
        frame = k+1
        print(frame)
        test_result_path = '../results/'+str(frame)+'/'

        if not os.path.exists(test_result_path):
            os.makedirs(test_result_path)

        K.clear_session()
        model_path = '../models/'+str(args.input)+'-'+str(frame)+str(args.model_no[k])+'h5'
        model = init_model(args.input)
        model.load_weights(model_path, by_name=True)

        test_results_seq(model, test_result_path, val_basic, val_next, k)
        r = show_results(test_result_path)
        

        
if __name__ == '__main__':
    main()