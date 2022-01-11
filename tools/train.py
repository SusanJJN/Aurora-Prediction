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
    parser.add_argument('--output',
                        help='output frames index',
                        required=True,
                        type=int)
    parser.add_argument('--epochs',
                        help='epoch num',
                        default=100,
                        type=int)
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    print(args.input)
    
    if args.input <= args.output:
        basic_frames = args.output
        interval_frames = args.input
    else:
        basic_frames = args.input
        interval_frames = args.output
    
    input_frames = args.input
    log_path = '../logs/'+str(args.input)+'-'+str(args.output)+'/'
    h5_path = '../models/'+str(args.input)+'-'+str(args.output)+'/'
    print('log_path:', log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(h5_path):
        os.makedirs(h5_path)

    tra_basic, tra_next = read_data(basic_frames, interval_frames, TRA_NPZ_PATH)
    val_basic, val_next = read_data(basic_frames, interval_frames, VAL_NPZ_PATH)

    np.random.seed(200)
    np.random.shuffle(tra_basic)
    np.random.seed(200)
    np.random.shuffle(tra_next)

    K.clear_session()
    model = utils.init_model(input_frames)
    training(model, tra_basic[:,:input_frames], val_basic[:,:input_frames], tra_next, val_next, log_path, args.epochs)


if __name__ == '__main__':
    main()
    