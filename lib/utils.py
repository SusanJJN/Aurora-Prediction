import os
import sys
sys.path.append("..")
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import ConvLSTM2D, Conv3D, Conv2D, Dense, Flatten, BatchNormalization, Input, LSTM, TimeDistributed, Conv2DTranspose, UpSampling2D, MaxPooling2D, merge, Reshape, Lambda
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from skimage.measure import compare_ssim, compare_psnr
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.metrics import mean_absolute_error
from keras.utils import np_utils, multi_gpu_model
from .config import *
from .dataset import *

img_rows = HEIGHT
img_cols = WIDTH
img_chns = IMG_CHNS
batch_size = BATCH_SIZE
tra_npz_path = TRA_NPZ_PATH
test_npz_path = VAL_NPZ_PATH


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.ssim = []
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.ssim.append(logs.get('ssim'))
        

def ssim(y_true, y_pred):
    score = tf.image.ssim(y_true, y_pred, 1.0)
    return score


def read_data(basic_frames, interval_frames, npz_path):

    npz_files = os.listdir(npz_path)
    npz_files.sort()

    for i in range(len(npz_files)):
        basic_seq, next_seq = get_3chn_seq(npz_path, npz_files[i], basic_frames, interval_frames, img_rows, img_chns)
        if i == 0:
            basic_images = basic_seq
            next_images = next_seq[:, -1, :, :, :]
        else:
            basic_images = np.vstack((basic_images, basic_seq))
            next_images = np.vstack((next_images, next_seq[:, -1, :, :, :]))

    basic_imgs = basic_images.reshape(basic_images.shape[0], basic_frames, img_rows, img_rows, img_chns)
    print(basic_imgs.shape, next_images.shape)

    return basic_imgs, next_images


def init_model(input_frames):
    filters = [32, 64, 128, 64, 32]
    images = Input(shape=(input_frames, img_rows, img_cols, img_chns), name='input_images')
    # print(images.shape)

    lstm_1 = ConvLSTM2D(filters=filters[0], kernel_size=(3, 3), strides=1, padding='same', return_sequences=True,
                        name='lstm_1')(images)
    pool_1 = TimeDistributed(MaxPooling2D(), name='pool_1')(lstm_1)
    bn_1 = BatchNormalization(name='bn_1')(pool_1)

    lstm_2 = ConvLSTM2D(filters=filters[1], kernel_size=(3, 3), strides=1, padding='same', return_sequences=True,
                        name='lstm_2')(bn_1)
    pool_2 = TimeDistributed(MaxPooling2D(), name='pool_2')(lstm_2)
    bn_2 = BatchNormalization(name='bn_2')(pool_2)

    lstm_3 = ConvLSTM2D(filters=filters[2], kernel_size=(3, 3), strides=1, padding='same', return_sequences=False,
                        name='lstm_3')(bn_2)
    # pool_3 = TimeDistributed(MaxPooling2D(), name='pool_3')(lstm_3)
    bn_3 = BatchNormalization(name='bn_3')(lstm_3)

    dec_1 = Conv2DTranspose(filters=filters[3], kernel_size=3, strides=1, padding='same', name='dec_1')(bn_3)
    pool_4 = UpSampling2D(name='pool_4')(dec_1)
    bn_4 = BatchNormalization(name='bn_4')(pool_4)

    dec_2 = Conv2DTranspose(filters=filters[4], kernel_size=3, strides=1, padding='same', name='dec_2')(bn_4)
    pool_5 = UpSampling2D(name='pool_5')(dec_2)
    bn_5 = BatchNormalization(name='bn_5')(pool_5)

    dec_3 = Conv2DTranspose(filters=img_chns, kernel_size=3, strides=1, activation='sigmoid', padding='same',
                            name='dec_3')(bn_5)
    output = dec_3

    model = Model(images, output)
    return model


def training(model, tra_basic, val_basic, tra_next, val_next, log_path, h5_path, epoch_num=10):

    opt = RMSprop(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[ssim])
    tensorboard = TensorBoard(log_dir=log_path)
    history = LossHistory()

    for i in range(epoch_num):
        print('epoch:', i)

        model.fit(tra_basic, tra_next, batch_size=8, epochs=1, validation_data=[val_basic, val_next],
                  callbacks=[history, tensorboard])
        model.save(h5_path + str(i) + '.h5')
        with open(log_path+'train_loss.txt', 'a') as f:
            f.write(str(history.losses)+'\n')

        with open(log_path+'train_ssim.txt', 'a') as f:
            f.write(str(history.ssim)+'\n')


def test_results_seq(model, result_path, val_basic, val_next, frame):
    for i in range(val_basic.shape[0]):
        gen_imgs = model.predict(val_basic[i:i + 1])

        true_sum = np.sum(val_next[i, frame, :, :, 0])
        pred_sum = np.sum(gen_imgs[0, :, :, 0])
        ssim = compare_ssim(val_next[i, frame, :, :, 0], gen_imgs[0, :, :, 0], data_range=1)

        mse = mean_squared_error(val_next[i, frame, :, :, 0], gen_imgs[0, :, :, 0])
        with open(result_path + 'true_sum.txt', 'a') as f:
            f.write(str(true_sum) + '\n')
        with open(result_path + 'pred_sum.txt', 'a') as f:
            f.write(str(pred_sum) + '\n')
        with open(result_path + 'ssim.txt', 'a') as f:
            f.write(str(ssim) + '\n')
        with open(result_path + 'mse.txt', 'a') as f:
            f.write(str(mse) + '\n')


def show_results(result_path):

    true_data = pd.read_csv(result_path + 'true_sum.txt', header=None)
    true_sum = true_data.values


    pred_data = pd.read_csv(result_path + 'pred_sum.txt', header=None)
    pred_sum = pred_data.values

    ssim_data = pd.read_csv(result_path + 'ssim.txt', header=None)
    ssim_list = ssim_data.values

    mse_data = pd.read_csv(result_path + 'mse.txt', header=None)
    mse_list = mse_data.values

    avg_true = np.mean(true_sum)
    avg_pred = np.mean(pred_sum)

    print('avg_true_sum:', avg_true, 'avg_pred_sum:', avg_pred)

    pred_mae = abs(avg_true - avg_pred)
    print('pred_mae:', pred_mae)

    avg_ssim = np.mean(ssim_list)
    print('avg_ssim:', avg_ssim)

    avg_mse = np.mean(mse_list)
    print('avg_mse:', avg_mse)

    return pred_mae, avg_ssim, avg_mse

def test_results_case(model, result_path, val_basic, val_next, frame):
    
    
    
    gen_imgs = model.predict(val_basic[i:i + 1])

    true_sum = np.sum(val_next[i, frame, :, :, 0])
    pred_sum = np.sum(gen_imgs[0, :, :, 0])
    ssim = compare_ssim(val_next[i, frame, :, :, 0], gen_imgs[0, :, :, 0], data_range=1)

    mse = mean_squared_error(val_next[i, frame, :, :, 0], gen_imgs[0, :, :, 0])
    with open(result_path + 'true_sum.txt', 'a') as f:
        f.write(str(true_sum) + '\n')
    with open(result_path + 'pred_sum.txt', 'a') as f:
        f.write(str(pred_sum) + '\n')
    with open(result_path + 'ssim.txt', 'a') as f:
        f.write(str(ssim) + '\n')
    with open(result_path + 'mse.txt', 'a') as f:
        f.write(str(mse) + '\n')

