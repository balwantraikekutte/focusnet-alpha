#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import keras
from keras.models import *
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, Activation
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import array_to_img

from se import squeeze_excite_block
from layers import initial_conv_block
from resnet import _conv_bn_relu, _residual_block, basic_block

from keras.preprocessing.image import flip_axis, random_channel_shift

from augmentation import random_rotation, random_zoom

import keras.backend as K

trainData = np.load('data.npy')
trainMask = np.load('dataMask.npy')

valData = np.load('dataval.npy')
valMask = np.load('dataMaskval.npy')

trainData = trainData.astype('float32')
mean = np.mean(trainData)  # mean for data centering
std = np.std(trainData)  # std for data normalization

valData = valData.astype('float32')

trainData -= mean
trainData /= std

valData -= mean 
valData /= std

trainMask = trainMask.astype('float32')
trainMask /= 255.  # scale masks to [0, 1]

valMask = valMask.astype('float32')
valMask /= 255.  # scale masks to [0, 1]

def Augmentation(X, Y):
        print('Augmentation model...')
        total = len(X)
        x_train, y_train = [], []
        
        for i in range(total):
            x, y = X[i], Y[i]
            #standart
            x_train.append(x)
            y_train.append(y)
        
#            for _ in xrange(1):
#                _x, _y = elastic_transform(x[0], y[0], 100, 20)
#                x_train.append(_x.reshape((1,) + _x.shape))
#                y_train.append(_y.reshape((1,) + _y.shape))
            
            #flip x
            x_train.append(flip_axis(x, 2))
            y_train.append(flip_axis(y, 2))
            #flip y
            x_train.append(flip_axis(x, 1))
            y_train.append(flip_axis(y, 1))
            #continue
            #zoom
            for _ in range(0):
                _x, _y = random_zoom(x, y, (0.9, 1.1))
                x_train.append(_x)
                y_train.append(_y)
            for _ in range(0):
                _x, _y = random_rotation(x, y, 5)
                x_train.append(_x)
                y_train.append(_y)
            #intentsity
            for _ in range(0):
                _x = random_channel_shift(x, 5.0)
                x_train.append(_x)
                y_train.append(y)
    
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train

x_train, y_train = Augmentation(trainData,trainMask)

smooth=1
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def get_unet():
    
    input = Input((256, 256, 3))
           
    conv1 = initial_conv_block(input) #512
    pool1 = _residual_block(basic_block, filters=32, repetitions=1, is_first_layer=False)(conv1) #256
    
    conv2 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=True)(pool1) #256
    pool2 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=False)(conv2) #128
    
    conv3 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=True)(pool2) #128
    pool3 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=False)(conv3) #64

    conv4 = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=True)(pool3) #64
    drop4 = Dropout(0.2)(conv4)
    
    up5 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4)) #128
    merge5 = keras.layers.Concatenate()([conv3,up5]) 
    conv5 = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=True)(merge5) #128
    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5)) #256
    merge6 = keras.layers.Concatenate()([conv2,up6])
    conv6 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=True)(merge6) #256
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6)) #512
    merge7 = keras.layers.Concatenate()([conv1,up7])
    conv7 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=True)(merge7) #512
    

    conv1r = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(1, 1))(input) #512
  
    block1 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=True)(conv1r) #512
    se1 = squeeze_excite_block(block1)
    gate1 = Activation('sigmoid')(conv7)
    block1concat = keras.layers.Multiply()([se1, gate1]) #512
    block1se = squeeze_excite_block(block1concat)
    block1b = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=False)(block1se) #256
    
    block2 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=True)(block1b) #256
    se2 = squeeze_excite_block(block2)
    gate2 = Activation('sigmoid')(conv6)
    block2concat = keras.layers.Multiply()([se2, gate2]) #256
    block2se = squeeze_excite_block(block2concat)
    block2b = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=False)(block2se) #128

    block3 = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=True)(block2b) #128
    se3 = squeeze_excite_block(block3)
    gate3 = Activation('sigmoid')(conv5)
    block3concat = keras.layers.Multiply()([se3, gate3]) #128
    block3se = squeeze_excite_block(block3concat)
    block3b = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=False)(block3se) # 64

    block4 = _residual_block(basic_block, filters=512, repetitions=1, is_first_layer=True)(block3b) #64
    block4se = squeeze_excite_block(block4)
    block4b = _residual_block(basic_block, filters=512, repetitions=1, is_first_layer=False)(block4se) #32

    up2_5 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(block4b)) #64
    merge2_5 = keras.layers.Concatenate()([block3b,up2_5])
    conv2_5 = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=True)(merge2_5) #64

    up2_6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2_5)) #128
    merge2_6 = keras.layers.Concatenate()([block2b,up2_6])
    conv2_6 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=True)(merge2_6) #128
    
    up2_7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2_6)) #256
    merge2_7 = keras.layers.Concatenate()([block1b,up2_7])
    conv2_7 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=True)(merge2_7) #256
         
    up2_8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2_7)) #512
    merge2_8 = keras.layers.Concatenate()([conv1r,up2_8])
    conv2_8 = _residual_block(basic_block, filters=32, repetitions=1, is_first_layer=True)(merge2_8) #512
    conv2_8 = _residual_block(basic_block, filters=16, repetitions=1, is_first_layer=True)(conv2_8)
    conv2_8 = _residual_block(basic_block, filters=4, repetitions=1, is_first_layer=True)(conv2_8)
         
    out = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv2_8)
         
    model = Model(inputs=input, outputs=out)

    model.compile(optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True), loss = dice_coef_loss, metrics = [dice_coef, jaccard_coef, 'acc'])
         
    model.summary()

    return model


model = get_unet()

model.load_weights('unetpolishedaug.h5')
         
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=0.01),
    ModelCheckpoint("unetpolishedaug.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=0.5e-7)
            ]

model.fit(x_train, y_train, batch_size=8, epochs=15, verbose=1,validation_data=(valData, valMask), shuffle=True, callbacks=callbacks)







