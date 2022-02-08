import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Add,Cropping2D
from tensorflow.keras import backend as K

name_list=['.1','.2','.3']

def Conv_Block(input,stage,strides,sub_name,name=name_list):
    x = Conv2D(64 * (2 ** (stage - 1)), kernel_size=1, strides=strides, padding='same',name='Conv_' + str(stage) + '.' + sub_name + name[0])(input)
    x = BatchNormalization(name="Batch_" + str(stage) + '.' + sub_name + name[0])(x)
    x = Activation('relu', name='Relu_' + str(stage) + '.' + sub_name + name[0])(x)
    x = Conv2D(64 * (2 ** (stage - 1)), kernel_size=3, strides=1, padding='same',name='Conv_' + str(stage) + '.' + sub_name + name[1])(x)
    x = BatchNormalization(name="Batch_" + str(stage) + '.' + sub_name + name[1])(x)
    x = Activation('relu', name='Relu_' + str(stage) + '.' + sub_name + name[1])(x)
    x = Conv2D(256 * (2 ** (stage - 1)), kernel_size=1, strides=1, padding='same',name='Conv_' + str(stage) + '.' + sub_name + name[2])(x)
    x = BatchNormalization(name="Batch_" + str(stage) + '.' + sub_name + name[2])(x)

    res = Conv2D(256 * (2 ** (stage - 1)), kernel_size=1, strides=strides, padding='same',name='Skip_conv_' + str(stage))(input)
    res = BatchNormalization(name='Skip_batch_' + str(stage))(res)

    out = Add()([res, x])
    out = Activation('relu', name='Out_relu_' + str(stage) + '.' + sub_name)(out)
    return out

def ID_Block(input, stage, sub_name, name=name_list):
    x = Conv2D(64 * (2 ** (stage - 1)), kernel_size=1, strides=1, padding='same',name='Conv_' + str(stage) + '.' + sub_name + name[0])(input)
    x = BatchNormalization(name='Batch_' + str(stage) + '.' + sub_name + name[0])(x)
    x = Activation('relu', name='Relu_' + str(stage) + '.' + sub_name + name[0])(x)
    x = Conv2D(64 * (2 ** (stage - 1)), kernel_size=3, strides=1, padding='same',name='Conv_' + str(stage) + '.' + sub_name + name[1])(x)
    x = BatchNormalization(name='Batch_' + str(stage) + '.' + sub_name + name[1])(x)
    x = Activation('relu', name='Relu_' + str(stage) + '.' + sub_name + name[1])(x)
    x = Conv2D(256 * (2 ** (stage - 1)), kernel_size=1, strides=1, padding='same',name='Conv_' + str(stage) + '.' + sub_name + name[2])(x)
    x = BatchNormalization(name='Batch_' + str(stage) + '.' + sub_name + name[2])(x)

    res = input

    out = Add()([res, x])
    out = Activation('relu', name='Out_relu_' + str(stage) + '.' + sub_name)(out)
    return out

def Get_layer(model, shape):
    # get the name of the first layer with shape in reverse direction
    name = None
    for layer in reversed(model.layers):
        if -1 < shape[1] - layer.output.shape[1] < 2:
            name = layer.output.name.split('/')[0]
            return name
            break
    if name == None:
        raise ValueError("Can't not find any layer with that shape")

class Crop(Layer):
    # layer that crop image to specified size
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape[1:3]

    def call(self, input, **kwargs):
        smaller_height, smaller_width = self.target_shape
        height, width = input.shape[1:3]
        if (height - smaller_height) / 2 == 0:
            top, bottom = (height - smaller_height) / 2
            left, right = (width - smaller_width) / 2
            crop = ((top, bottom), (left, right))
        else:
            top = int((height - smaller_height) / 2)
            bottom = int((height - smaller_height) / 2) + int((height - smaller_height) % 2)
            left = int((width - smaller_width) / 2)
            right = int((width - smaller_width) / 2) + int((width - smaller_width) % 2)
            crop = ((top, bottom), (left, right))
        return Cropping2D(cropping=crop)(input)

class Resize(Layer):
    # resize layer
    def __init__(self,shape,**kwargs):
        super().__init__(**kwargs)
        self.shape=shape

    def call(self, input, **kwargs):
        return tf.image.resize(input, self.shape, method=tf.image.ResizeMethod.BILINEAR)

class MeanVar_Norm(Layer):
    # normalize input layer
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input, epsilon=1e-7, **kwargs):
        mean = K.mean(input)
        std = K.std(input)
        return (input - mean) / (std + epsilon)