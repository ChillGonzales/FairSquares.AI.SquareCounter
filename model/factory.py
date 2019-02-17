import keras
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

def create_model(model_name="inception"):
    if (model_name == "inception"):
        # Keras model
        base_model = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                weights='imagenet',
                                                                input_tensor=None,
                                                                input_shape=(299, 299, 3),
                                                                pooling=None)
    elif (model_name=="vgg19"):
        base_model = keras.applications.vgg19.VGG19(include_top=False, 
                                                    weights='imagenet', 
                                                    input_tensor=None, 
                                                    input_shape=(299, 299, 3), 
                                                    pooling=None)
    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    prediction = Dense(1, activation='linear')(x)
    head_model = Model(inputs=base_model.input, outputs=prediction)
    opt = Adam(lr = 0.0005)
    head_model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    return head_model