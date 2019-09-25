import keras
from keras.layers import Flatten, Dense, Dropout, Concatenate, Input, Conv2D, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam

def create_model(model_name: str = "strided", image_shape: tuple = (299, 299, 3), features_shape: tuple = (38, )):
    if (model_name == "inception"):
        # Keras model
        base_model = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                 weights='imagenet',
                                                                 input_tensor=None,
                                                                 input_shape=image_shape,
                                                                 pooling=None)
    elif (model_name == "vgg19"):
        base_model = keras.applications.vgg19.VGG19(include_top=False,
                                                    weights='imagenet',
                                                    input_tensor=None,
                                                    input_shape=image_shape,
                                                    pooling=None)
    elif (model_name == "strided"):
        # Source: https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
        base_model = Sequential()
        reg = l2(0.5)
        init = "he_normal"
        chanDim = -1
        cnnDropout = 0.25
        denseDropout = 0.5
        base_model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="valid",
                              kernel_initializer=init, kernel_regularizer=reg,
                              input_shape=image_shape))

        # here we stack two CONV layers on top of each other where
        # each layers will learn a total of 32 (3x3) filters
        base_model.add(Conv2D(32, (3, 3), padding="same",
                              kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
                                  kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Dropout(cnnDropout))

        # stack two more CONV layers, keeping the size of each filter
        # as 3x3 but increasing to 64 total learned filters
        base_model.add(Conv2D(64, (3, 3), padding="same",
                                kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
                                kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Dropout(cnnDropout))

        # increase the number of filters again, this time to 128
        base_model.add(Conv2D(128, (3, 3), padding="same",
                                kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
                                kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Dropout(cnnDropout))

    x = Flatten()(base_model.output)
    x = Dropout(denseDropout)(x)
    feature_input = Input(shape=features_shape)
    x = Concatenate()([x, feature_input])
    x = Dropout(denseDropout)(x)
    denseLayers = 1
    denseNeuronCount = 175
    m = Dense(denseNeuronCount)(x)
    b = Dense(denseNeuronCount)(x)
    for i in range(denseLayers):
        m = Dense(denseNeuronCount, activation='relu', kernel_initializer=init, kernel_regularizer=reg)(m)
        m = Dropout(denseDropout)(m)
        b = Dense(denseNeuronCount, activation='relu', kernel_initializer=init, kernel_regularizer=reg)(b)
        b = Dropout(denseDropout)(b)
    m = Dense(1)(m)
    m = Activation("linear", name="slope_output")(m)
    b = Dense(1)(b)
    b = Activation("linear", name="intercept_output")(b)
    head_model = Model(inputs=[base_model.input, feature_input], outputs=[m, b])
    return head_model