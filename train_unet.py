from __future__ import print_function
import os, glob
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import cv2
import settings

working_path = settings.BASE_DIR + 'tutorial_out/masks/'

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 320
img_cols = 320

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    
    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=[dice_coef])

    return model
    
def scheduler(epoch):
    if epoch < 5:
        return 0.005
    if epoch < 10:
        return 0.001
    if epoch < 20:
        return 0.0001
    return 1e-5

def get_data(data_dir):
    file_names = glob.glob(data_dir + '/train/*.jpg')
    x_train = [None] * len(file_names)
    for i, filename in enumerate(file_names):
        x_train[i] = cv2.imread(filename)

def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load(working_path+"train_final_images.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path+"train_final_masks.npy").astype(np.float32)

    imgs_val = np.load(working_path+"val_final_images.npy").astype(np.float32)
    imgs_mask_val = np.load(working_path+"val_final_masks.npy").astype(np.float32)

    #imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
    #imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)
    
    #mean = np.mean(imgs_train)  # mean for data centering
    #std = np.std(imgs_train)  # std for data normalization

    #minvalue = np.min(imgs_train)
    #maxvalue = np.max(imgs_train)
    #imgs_train -= minvalue
    #imgs_train /= (maxvalue-minvalue)

    #imgs_val -= minvalue
    #imgs_val /= (maxvalue-minvalue)

    imgs_mask_train *= 255
    imgs_mask_val *= 255

    print(imgs_train.shape[:2]+(320,320,))
    imgs_train_new = np.zeros(imgs_train.shape[:2]+(320,320,))
    imgs_val_new = np.zeros(imgs_val.shape[:2]+(320,320,))
    imgs_mask_train_new = np.zeros(imgs_mask_train.shape[:2]+(320,320,))
    imgs_mask_val_new = np.zeros(imgs_mask_val.shape[:2]+(320,320,))

    for i, img in enumerate(imgs_train):
        imgs_train_new[i][0] = cv2.resize(imgs_train[i][0], (320,320))
    for i, img in enumerate(imgs_val):
        imgs_val_new[i][0] = cv2.resize(imgs_val[i][0], (320,320))
    for i, img in enumerate(imgs_mask_train):
        imgs_mask_train_new[i][0] = cv2.resize(imgs_mask_train[i][0], (320,320))
    for i, img in enumerate(imgs_mask_val):
        imgs_mask_val_new[i][0] = cv2.resize(imgs_mask_val[i][0], (320,320))

    #imgs_train -= mean  # images should already be standardized, but just in case
    #imgs_train /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # Saving weights to unet.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        print('loading weight...')
        model.load_weights('./unet.hdf5')
        
    # 
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970 
    # I was able to run 20 epochs with a training set size of 320 and 
    # batch size of 2 in about an hour. I started getting reseasonable masks 
    # after about 3 hours of training.  


    lr_decay = LearningRateScheduler(scheduler)

    #
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train_new, imgs_mask_train_new, batch_size=16, epochs=100, verbose=1, shuffle=True,
                validation_data=(imgs_val_new, imgs_mask_val_new), callbacks=[model_checkpoint, lr_decay])

    # loading best weights from training session
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('./unet.hdf5')
'''
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    np.save('masksTestPredicted.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)
'''
if __name__ == '__main__':
    train_and_predict(True)
