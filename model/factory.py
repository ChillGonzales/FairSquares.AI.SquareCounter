import keras
from keras.layers import Flatten, Dense, Dropout, Concatenate, Input
from keras.models import Model
from keras.optimizers import Adam

def create_model(model_name="inception", image_shape=(299, 299, 3), features_shape=(38, )):
    if (model_name == "inception"):
        # Keras model
        base_model = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                weights='imagenet',
                                                                input_tensor=None,
                                                                input_shape=image_shape,
                                                                pooling=None)
    elif (model_name=="vgg19"):
        base_model = keras.applications.vgg19.VGG19(include_top=False, 
                                                    weights='imagenet', 
                                                    input_tensor=None, 
                                                    input_shape=image_shape,
                                                    pooling=None)
    x = Flatten()(base_model.output)
    feature_input = Input(shape=features_shape)
    x = Concatenate()([x, feature_input])
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    prediction = Dense(1, activation='linear')(x)
    head_model = Model(inputs=[base_model.input, feature_input], outputs=prediction)
    opt = Adam(lr = 0.00001)
    head_model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    return head_model