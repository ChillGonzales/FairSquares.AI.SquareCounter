import keras
from keras.layers import Flatten, Dense, Dropout, Concatenate, Input, Conv2D, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam


def create_model(model_name="inception", image_shape=(299, 299, 3), features_shape=(38, )):
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
        dropoutRate = 0.5
        # our first CONV layer will learn a total of 16 filters, each
        # Of which are 7x7 -- we'll then apply 2x2 strides to reduce
        # the spatial dimensions of the volume
        base_model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",
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
        base_model.add(Dropout(dropoutRate))

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
        base_model.add(Dropout(dropoutRate))

        # increase the number of filters again, this time to 128
        base_model.add(Conv2D(128, (3, 3), padding="same",
                                kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
                                kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Dropout(dropoutRate))

        base_model.add(Conv2D(128, (3, 3), padding="same",
                                kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
                                kernel_initializer=init, kernel_regularizer=reg))
        base_model.add(Activation("relu"))
        base_model.add(BatchNormalization(axis=chanDim))
        base_model.add(Dropout(dropoutRate))

    x = Flatten()(base_model.output)
    feature_input = Input(shape=features_shape)
    x = Concatenate()([x, feature_input])
    denseLayers = 5
    denseNeuronCount = 512
    for i in range(denseLayers):
        x = Dense(denseNeuronCount, activation='relu', kernel_initializer=init, kernel_regularizer=reg, bias_initializer=init, bias_regularizer=reg)(x)
        x = Dropout(dropoutRate)(x)
    x = Dense(128, activation='relu', kernel_initializer=init, kernel_regularizer=reg, bias_regularizer=reg, bias_initializer=init)(x)
    prediction = Dense(2, activation='relu')(x)
    head_model = Model(inputs=[base_model.input, feature_input], outputs=prediction)
    return head_model