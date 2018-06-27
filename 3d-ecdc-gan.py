from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers import Reshape
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

def discriminator(phase_train = True, params={'cube_size':32, 'strides':(2,2,2), 'kernel_size':(5,5,5), 'leak':0.2}):
## discriminator, input: 32x32x32 cube, output: Real or Fake.

    cube_size = params['cube_size']
    strides = params['strides']
    kernel_size = params['kernel_size'] 
    leak = params['leak']
    
    inputs = Input(shape=(cube_size, cube_size, cube_size, 1))

    d1 = Conv3D(filters=64, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(inputs)
    d1 = BatchNormalization()(d1, training=phase_train)
    d1 = LeakyReLU(leak)(d1)

    d2 = Conv3D(filters=128, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d1)
    d2 = BatchNormalization()(d2, training=phase_train)
    d2 = LeakyReLU(leak)(d2)

    d3 = Conv3D(filters=256, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d2)
    d3 = BatchNormalization()(d3, training=phase_train)
    d3 = LeakyReLU(leak)(d3)

    d4 = Conv3D(filters=1, kernel_size=kernel_size,
                  strides=(1, 1, 1), kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='valid')(d3)
    
    d4 = BatchNormalization()(d4, training=phase_train)

    d4 = Activation(activation='sigmoid')(d4) 

    model = Model(inputs=inputs, outputs=d4)
    model.summary()

    return model


def decoder(phase_train=True, params={'z_length':200, 'strides':(2,2,2), 'kernel_size':(5,5,5)}):

    z_length = params['z_length']
    strides = params['strides']
    kernel_size = params['kernel_size'] 
    
    inputs = Input(shape=(1, 1, 1, z_length))

    dc1= Deconv3D(filters=256, kernel_size=kernel_size,
                  strides=(1, 1, 1), kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='valid')(inputs)
    dc1= BatchNormalization()(dc1,training=phase_train)
    dc1 = Activation(activation='relu')(dc1)    

    dc2 = Deconv3D(filters=128, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(dc1)
    dc2 = BatchNormalization()(dc2, training=phase_train)
    dc2 = Activation(activation='relu')(dc2)

    dc3 = Deconv3D(filters=64, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(dc2)
    dc3 = BatchNormalization()(dc3, training=phase_train)
    dc3 = Activation(activation='relu')(dc3)

    dc4 = Deconv3D(filters=1, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(dc3)
    dc4 = BatchNormalization()(dc4, training=phase_train)
    dc4 = Activation(activation='sigmoid')(dc4) 

    model = Model(inputs=inputs, outputs=dc4)
    model.summary()

    return model


def encoder(phase_train = True, params={'cube_size':32, 'strides':(2,2,2), 'kernel_size':(5,5,5), 'leak':0.2}):

    cube_size = params['cube_size']
    strides = params['strides']
    kernel_size = params['kernel_size'] 
    leak = params['leak']
    
    inputs = Input(shape=(cube_size, cube_size, cube_size, 1))

    e1 = Conv3D(filters=64, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(inputs)
    e1 = BatchNormalization()(e1, training=phase_train)
    e1 = LeakyReLU(leak)(e1)

    e2 = Conv3D(filters=128, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d1)
    e2 = BatchNormalization()(e2, training=phase_train)
    e2 = LeakyReLU(leak)(e2)

    e3 = Conv3D(filters=256, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d2)
    e3 = BatchNormalization()(e3, training=phase_train)
    e3 = LeakyReLU(leak)(e3)

    e4 = Reshape(1, 1, 1, z_length)(d3)

    model = Model(inputs=inputs, outputs=e4)
    model.summary()

    return model