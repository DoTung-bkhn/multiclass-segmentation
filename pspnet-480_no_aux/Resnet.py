from BlockBuilder import*
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D

def Resnet50(input, resize_factor,normalize_input=True):
    if normalize_input:
        x = MeanVar_Norm(name='Normalize_Input')(input)
        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='Start_conv')(x)
    else:
        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='Start_conv')(input)
    x = BatchNormalization(name='Start_batch')(x)
    x = Activation('relu', name='Start_relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='Max_pool')(x)

    resize_factor *= 1 / 4

    model_config = [3, 4, 6, 3]
    for i in range(len(model_config)):
        if resize_factor != 1:
            resize_factor *= 1 / 2
            strides = 2
        else:strides = 1
        x = Conv_Block(x, stage=i + 1, strides=strides, sub_name='1')
        for block in range(model_config[i] - 1):
            x = ID_Block(x, stage=i + 1, sub_name=str(block + 2))
    model=Model(inputs=input,outputs=x,name='Resnet50')
    return model

def Resnet101(input,resize_factor,normalize_input=True):
    if normalize_input:
        x = MeanVar_Norm(name='Normalize_Input')(input)
        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='Start_conv')(x)
    else:
        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='Start_conv')(input)
    x = BatchNormalization(name='Start_batch')(x)
    x = Activation('relu', name='Start_relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='Max_pool')(x)

    resize_factor *= 1 / 4

    model_config = [3, 4, 23, 3]
    for i in range(len(model_config)):
        if resize_factor != 1:
            resize_factor *= 1 / 2
            strides = 2
        else:
            strides = 1
        x = Conv_Block(x, stage=i + 1, strides=strides, sub_name='1')
        for block in range(model_config[i] - 1):
            x = ID_Block(x, stage=i + 1, sub_name=str(block + 2))

    model = Model(inputs=input, outputs=x, name='Resnet101')
    return model
