from Resnet import Resize
from tensorflow.keras.models import Model
from BlockBuilder import Crop,Get_layer
from tensorflow.keras.layers import AveragePooling2D,Activation,BatchNormalization,Concatenate,Conv2DTranspose,Conv2D,Dropout,Input

def Psp_net(input_shape,numb_class,encoder,resize_factor,bin_size,using_pretrained=False,pretrained_weights=None,weights=None):
    input = Input(shape=input_shape, name='Input')
    Encoder = encoder(input=input, resize_factor=resize_factor)
    if using_pretrained:
        Encoder.load_weights(pretrained_weights)
        Encoder.trainable = False
    feature_map = Encoder.output
    feature_mapSize = feature_map.shape[1:3]
    channel = feature_map.shape[-1]

    #pyramid pooling module
    for size in bin_size:
        if size > feature_mapSize[0]:
            raise ValueError('Incorrect bin size')
        break
    kernel = [feature_mapSize[0] if size == 1 else int(feature_mapSize[0] / size) + feature_mapSize[0] % size for size in bin_size]
    strides = [1 if size == 1 else int(feature_mapSize[0] / size) for size in bin_size]

    out = feature_map
    for i in range(len(bin_size)):
        x = AveragePooling2D(pool_size=kernel[i], strides=strides[i], name='Avg_pool_' + str(i + 1))(feature_map)
        x = Conv2D(int(channel / len(bin_size)), kernel_size=1, strides=1, name='Conv_' + str(i + 1))(x)
        x = BatchNormalization(name='Batch_' + str(i + 1))(x)
        x = Activation('relu', name='Relu_' + str(i + 1))(x)
        x = Resize(feature_mapSize, name='Resize_' + str(i + 1))(x)
        out = Concatenate()([out, x])

    # decoder
    out = Conv2D(numb_class, (3, 3), strides=(1, 1), padding="same")(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(0.3, name='Drop_out')(out)
    out = Conv2D(numb_class, (1, 1), strides=(1, 1), name='Final_conv')(out)
    out = Resize(input_shape[0:2], name='Final_resize')(out)
    out = Activation('softmax', name='Sofmax')(out)

    model = Model(inputs=input, outputs=out, name='Psp_net')
    if weights is not None:
        model.load_weights(weights)
    return model

def Unet(input_shape,numb_class,encoder,resize_factor,using_pretrained=False,pretrained_weights=None,weights=None):
    Encoder = encoder(input_shape=input_shape, resize_factor=resize_factor)
    if using_pretrained:
        Encoder.load_weights(pretrained_weights)
        Encoder.trainable = False
    feature_map = Encoder.output

    #decoder
    upsample1 = Conv2DTranspose(512, kernel_size=3, strides=2, padding='same', activation='relu', name='Upsample_1')(feature_map)
    layer = Encoder.get_layer(Get_layer(Encoder, upsample1.shape)).output
    if layer.shape[1] != upsample1.shape[1]:
        crop1 = Crop(layer.shape, name='Crop_1')(upsample1)
        merge1 = Concatenate(name='Merge_1')([crop1, layer])
    else:
        merge1 = Concatenate(name='Merge_1')([upsample1, layer])
    de_conv1 = Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu', name='De_conv1')(merge1)
    batch1 = BatchNormalization(name='Batch_1')(de_conv1)
    de_conv2 = Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu', name='De_conv2')(batch1)
    batch2 = BatchNormalization(name='Batch_2')(de_conv2)

    upsample2 = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same', activation='relu', name='Upsample_2')(batch2)
    layer = Encoder.get_layer(Get_layer(Encoder, upsample2.shape)).output
    if layer.shape[1] != upsample2.shape[1]:
        crop2 = Crop(layer.shape, name='Crop_2')(upsample2)
        merge2 = Concatenate(name='Merge_2')([crop2, layer])
    else:
        merge2 = Concatenate(name='Merge_2')([upsample2, layer])
    de_conv3 = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu', name='De_conv3')(merge2)
    batch3 = BatchNormalization(name='Batch_3')(de_conv3)
    de_conv4 = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu', name='De_conv4')(batch3)
    batch4 = BatchNormalization(name='Batch_4')(de_conv4)

    upsample3 = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu', name='Upsample_3')(batch4)
    layer = Encoder.get_layer(Get_layer(Encoder, upsample3.shape)).output
    if layer.shape[1] != upsample3.shape[1]:
        crop3 = Crop(layer.shape, name='Crop_3')(upsample3)
        merge3 = Concatenate(name='Merge_3')([crop3, layer])
    else:
        merge3 = Concatenate(name='Merge_3')([upsample3, layer])
    de_conv5 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu', name='De_conv5')(merge3)
    batch5 = BatchNormalization(name='Batch_5')(de_conv5)
    de_conv6 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu', name='De_conv6')(batch5)
    batch6 = BatchNormalization(name='Batch_6')(de_conv6)

    upsample4 = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu', name='Upsample_4')(batch6)
    layer = Encoder.get_layer(Get_layer(Encoder, upsample4.shape)).output
    if layer.shape[1] != upsample4.shape[1]:
        crop4 = Crop(layer.shape, name='Crop_4')(upsample4)
        merge4 = Concatenate(name='Merge_4')([crop4, layer])
    else:
        merge4 = Concatenate(name='Merge_4')([upsample4, layer])
    de_conv7 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', name='De_conv7')(merge4)
    batch7 = BatchNormalization(name='Batch_7')(de_conv7)
    de_conv8 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', name='De_conv8')(batch7)
    batch8 = BatchNormalization(name='Batch_8')(de_conv8)

    upsample5 = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu', name='Upsample5')(batch8)

    pred = Conv2D(numb_class, kernel_size=1, strides=1, padding='same', activation='softmax', name='Predict')(upsample5)

    model = Model(inputs=Encoder.input, outputs=pred, name='Unet')
    if weights is not None:
        model.load_weights(weights)
    return model

