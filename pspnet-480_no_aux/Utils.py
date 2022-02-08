import os
import cv2 as cv
import numpy as np
import random as rand
import matplotlib.pyplot as plt

def load_img(path,output_shape,number=0):
    img_set = []
    anno_set = []
    for subfile in os.listdir(path):
        if 'labels' not in subfile:
            img_file = subfile
        if 'labels' in subfile:
            anno_file = subfile
    for img in os.listdir(os.path.join(path, img_file)):
        img_name = img
        anno_name = img.split('.')[0] + '_L.png'
        img_fullname = os.path.join(path, img_file, img_name)
        anno_fullname = os.path.join(path, anno_file, anno_name)
        img = cv.imread(img_fullname)
        img = cv.resize(img, output_shape)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        anno = cv.imread(anno_fullname)
        anno = cv.resize(anno, output_shape)
        anno = cv.cvtColor(anno, cv.COLOR_BGR2RGB)
        img_set.append(img)
        anno_set.append(anno)
    img_set = np.asarray(img_set, dtype='int32')
    anno_set = np.asarray(anno_set, dtype='int32')
    if number != 0:
        Range = np.arange(len(img_set))
        for i in range(number):
            index = rand.choice(Range)
            plt.figure(i + 1, figsize=(10, 10))
            plt.subplot(1, 2, 1).imshow(img_set[index])
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('Original Image', fontsize=15)
            plt.subplot(1, 2, 2).imshow(anno_set[index])
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('Corresponding Annotation', fontsize=15)
            plt.show()
    return img_set, anno_set

def Visualize_colormap(Dataset_class):
    numb_class = len(Dataset_class)
    name = [label.name for label in Dataset_class]
    color = [label.color for label in Dataset_class]
    back_ground = np.full((480, 450, 3), 255, dtype='uint8')
    for i in range(numb_class):
        map = cv.rectangle(back_ground, (0, 0 + 15 * i), (300, 15 + 15 * i), color[i], thickness=-1)
        cv.putText(map, name[i], (305, 10 + 15 * i), cv.FONT_HERSHEY_SIMPLEX, 0.5, color[i], thickness=1)
        map = cv.resize(map, (480, 480))
    cv.imshow('Color map', cv.cvtColor(map, cv.COLOR_BGR2RGB))
    cv.imwrite('Color_map.png', cv.cvtColor(map, cv.COLOR_BGR2RGB))
    cv.waitKey(0)
    cv.destroyAllWindows()

def OnehotEncode(anno_list, color_map):
    anno_map_list = np.zeros((len(anno_list), anno_list.shape[1], anno_list.shape[2], len(color_map)), dtype='uint8')
    for i, mask in enumerate(anno_list):
        for j, color in enumerate(color_map):
            check = np.all(mask.reshape(-1, 3) == color_map[j].color, axis=1).reshape(mask.shape[:2])
            anno_map_list[i, :, :, j] = check
    return anno_map_list.astype('float32') * 1

def OnehotDecode(mask_map, color_map):
    rgb = np.zeros(shape=(mask_map.shape[0], mask_map.shape[1], 3))
    max_array = np.argmax(mask_map, axis=-1)
    for i in range(len(color_map)):
        rgb[max_array == i] = color_map[i].color
    return rgb.astype('uint8')

def Plotting_metrics(metrics_list,metrics_name):
    train = metrics_list[metrics_name]
    val = metrics_list['val_' + metrics_name]
    epoch = len(train)
    x = np.linspace(1, epoch, epoch)
    plt.figure(figsize=(5, 3), dpi=110)
    plt.plot(x, train, 'r', label='train')
    plt.plot(x, val, 'b', label='validation')
    plt.legend()
    plt.yticks([])
    plt.ylabel(metrics_name)
    plt.xlabel('Epoch')
    plt.show()

def Predict(model, image, color_map):
    height, width, channel = image.shape
    image = image.reshape(-1, height, width, channel)
    pred = model.predict(image)
    pred = pred.reshape(height, width, pred.shape[-1])
    rgb = OnehotDecode(pred, color_map)
    return rgb
