#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:18:24 2018

@author: ck807
"""
import keras

from keras import layers

import keras.backend as K

from keras.regularizers import l2

from Batch_Normalization import BatchNormalization


ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

def initial_conv_block1(input, weight_decay=5e-4):
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), name='conv1_1')(input)
    x = BatchNormalization(axis=CHANNEL_AXIS, name='BN1_1', freeze=False)(x)
    x = layers.LeakyReLU(name='Activation1_1')(x)
    return x



def conv_block2(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN2_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation2_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv2_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN2_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation2_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv2_2')(prev) 
    return prev
def skip_block2(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv2_3')(prev)
    return prev 
def Residual2(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block2(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block2(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add2_1')([skip, conv]) # the residual connection



def conv_block3(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN3_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation3_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv3_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN3_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation3_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv3_2')(prev) 
    return prev
def skip_block3(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv3_3')(prev)
    return prev 
def Residual3(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block3(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block3(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add3_1')([skip, conv]) # the residual connection



def conv_block4(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN4_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation4_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv4_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN4_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation4_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv4_2')(prev) 
    return prev
def skip_block4(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv4_3')(prev)
    return prev 
def Residual4(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block4(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block4(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add4_1')([skip, conv]) # the residual connection



def conv_block5(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN5_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation5_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv5_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN5_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation5_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv5_2')(prev) 
    return prev
def skip_block5(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv5_3')(prev)
    return prev 
def Residual5(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block5(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block5(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add5_1')([skip, conv]) # the residual connection



def conv_block6(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN6_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation6_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv6_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN6_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation6_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv6_2')(prev) 
    return prev
def skip_block6(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv6_3')(prev)
    return prev 
def Residual6(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block6(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block6(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add6_1')([skip, conv]) # the residual connection



def conv_block7(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN7_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation7_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv7_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN7_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation7_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv7_2')(prev) 
    return prev
def skip_block7(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv7_3')(prev)
    return prev 
def Residual7(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block7(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block7(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add7_1')([skip, conv]) # the residual connection



def conv_block8(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN8_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation8_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv8_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN8_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation8_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv8_2')(prev) 
    return prev
def skip_block8(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv8_3')(prev)
    return prev 
def Residual8(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block8(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block8(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add8_1')([skip, conv]) # the residual connection



def conv_block9(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN9_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation9_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv9_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN9_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation9_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv9_2')(prev) 
    return prev
def skip_block9(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv9_3')(prev)
    return prev 
def Residual9(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block9(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block9(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add9_1')([skip, conv]) # the residual connection



def conv_block10(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN10_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation10_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv10_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN10_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation10_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv10_2')(prev) 
    return prev
def skip_block10(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv10_3')(prev)
    return prev 
def Residual10(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block10(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block10(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add10_1')([skip, conv]) # the residual connection



def conv_block11(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN11_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation11_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv11_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN11_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation11_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv11_2')(prev) 
    return prev
def skip_block11(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv11_3')(prev)
    return prev 
def Residual11(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block11(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block11(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add11_1')([skip, conv]) # the residual connection



def conv_block12(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN12_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation12_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv12_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN12_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation12_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv12_2')(prev) 
    return prev
def skip_block12(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv12_3')(prev)
    return prev 
def Residual12(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block12(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block12(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add12_1')([skip, conv]) # the residual connection



def conv_block13(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN13_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation13_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv13_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN13_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation13_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv13_2')(prev) 
    return prev
def skip_block13(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv13_3')(prev)
    return prev 
def Residual13(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block13(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block13(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add13_1')([skip, conv]) # the residual connection



def conv_block14(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN14_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation14_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv14_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN14_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation14_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv14_2')(prev) 
    return prev
def skip_block14(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv14_3')(prev)
    return prev 
def Residual14(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block14(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block14(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add14_1')([skip, conv]) # the residual connection



def conv_block15(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN15_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation15_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv15_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN15_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation15_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv15_2')(prev) 
    return prev
def skip_block15(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv15_3')(prev)
    return prev 
def Residual15(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block15(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block15(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add15_1')([skip, conv]) # the residual connection



def conv_block16(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN16_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation16_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv16_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN16_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation16_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv16_2')(prev) 
    return prev
def skip_block16(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv16_3')(prev)
    return prev 
def Residual16(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block16(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block16(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add16_1')([skip, conv]) # the residual connection


def conv_block17(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN17_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation17_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv17_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN17_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation17_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv17_2')(prev) 
    return prev
def skip_block17(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv17_3')(prev)
    return prev 
def Residual17(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block17(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block17(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add17_1')([skip, conv]) # the residual connection



def conv_block18(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN18_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation18_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv18_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN18_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation18_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv18_2')(prev) 
    return prev
def skip_block18(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv18_3')(prev)
    return prev 
def Residual18(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block18(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block18(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add18_1')([skip, conv]) # the residual connection



def conv_block19(feat_maps_out, prev):
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN19_1', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation19_1')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv19_1')(prev) 
    prev = BatchNormalization(axis=CHANNEL_AXIS, name='BN19_2', freeze=False)(prev) # Specifying the axis and mode allows for later merging
    prev = layers.LeakyReLU(name='Activation19_2')(prev)
    prev = layers.Conv2D(feat_maps_out, (3,3), padding = 'same', kernel_initializer = 'he_normal', name='Conv19_2')(prev) 
    return prev
def skip_block19(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out, (1,1), padding = 'same', kernel_initializer = 'he_normal', name='Conv19_3')(prev)
    return prev 
def Residual19(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block19(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block19(feat_maps_out, prev_layer)
    return keras.layers.Add(name='Add19_1')([skip, conv]) # the residual connection



