#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:38:01 2019

@author: chaitanya
"""

from keras import layers
from keras import models


#
# image dimensions
#

img_height = 256
img_width = 256
img_channels = 3

#
# network params
#

cardinality = 3


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
        
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y
    
    def attention_block(y, nb_channels_in, _strides):
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y_scores = layers.Activation('softmax')(y)
        y = add_common_layers(y)
        
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = layers.Multiply()([y_scores, y])
        y = add_common_layers(y)
        
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = add_common_layers(y)
        
        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(attention_block(group, _d, _strides))
            #groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y
    
    conv1 = layers.Conv2D(84, kernel_size=(3, 3), strides=(1,1), padding='same')(x)
    conv1_d = residual_block(conv1, 84, 84, _project_shortcut=False, _strides=(2,2))
    
    conv2 = residual_block(conv1_d, 84, 144, _project_shortcut=True, _strides=(1,1))
    conv2_d = residual_block(conv2, 144, 144, _project_shortcut=False, _strides=(2,2))
    
    conv3 = residual_block(conv2_d, 144, 255, _project_shortcut=True, _strides=(1,1))
    conv3_d = residual_block(conv3, 255, 255, _project_shortcut=False, _strides=(2,2))
    
    conv4 = residual_block(conv3_d, 255, 396, _project_shortcut=True, _strides=(1,1))
    conv4_d = residual_block(conv4, 396, 396, _project_shortcut=False, _strides=(2,2))
    
    bottleneck = residual_block(conv4_d, 396, 510, _project_shortcut=True, _strides=(1,1))
    
    up1 = layers.UpSampling2D(size = (2,2))(bottleneck)
    up1_c = residual_block(up1, 510, 396, _project_shortcut=True, _strides=(1,1))
    merge1 = layers.Add()([conv4, up1_c])
    conv5 = residual_block(merge1, 396, 396, _project_shortcut=False, _strides=(1,1))
    
    up2 = layers.UpSampling2D(size = (2,2))(conv5)
    up2_c = residual_block(up2, 396, 255, _project_shortcut=True, _strides=(1,1))
    merge2 = layers.Add()([conv3, up2_c])
    conv6 = residual_block(merge2, 255, 255, _project_shortcut=False, _strides=(1,1))
    
    up3 = layers.UpSampling2D(size = (2,2))(conv6)
    up3_c = residual_block(up3, 255, 144, _project_shortcut=True, _strides=(1,1))
    merge3 = layers.Add()([conv2, up3_c])
    conv7 = residual_block(merge3, 144, 144, _project_shortcut=False, _strides=(1,1))
    
    up4 = layers.UpSampling2D(size = (2,2))(conv7)
    up4_c = residual_block(up4, 144, 84, _project_shortcut=True, _strides=(1,1))
    merge4 = layers.Add()([conv1, up4_c])
    conv8 = residual_block(merge4, 84, 48, _project_shortcut=True, _strides=(1,1))
    conv9 = residual_block(conv8, 48, 27, _project_shortcut=True, _strides=(1,1))
    conv10 = residual_block(conv9, 27, 9, _project_shortcut=True, _strides=(1,1))
    out = layers.Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    
    return out
        
image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = residual_network(image_tensor)
        
model = models.Model(inputs=[image_tensor], outputs=[network_output])
print(model.summary())
    

########################################################################################################################################################################
        
    
import numpy as np
import scipy.ndimage as ndi
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K

print("Loading data..")

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

print("Augmenting data..")

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def random_zoom(x, y, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def random_rotation(x, y, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y

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
#            for _ in range(0):
#                _x = random_channel_shift(x, 5.0)
#                x_train.append(_x)
#                y_train.append(y)
    
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train

x_train, y_train = Augmentation(trainData,trainMask)


########################################################################################################################################################################


smooth=1
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


########################################################################################################################################################################

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=0.01),
    ModelCheckpoint("unetpolishedaug.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=0.5e-7)
            ]
model.compile(optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True), loss = dice_coef_loss, metrics = [dice_coef, jaccard_coef, 'acc'])
model.fit(x_train, y_train, batch_size=8, epochs=30, verbose=1, validation_data=(valData, valMask), shuffle=True, callbacks=callbacks)


########################################################################################################################################################################
