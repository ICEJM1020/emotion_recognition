import pandas as pd
import numpy as np
import keras
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam


def brighter(ori, gamma=0.4, a=1.5, b=20, type=0):
    len = ori.size
    dst = np.zeros(len)
    if type==0:
        # 非线性变换
        color = np.clip(pow(ori / 255.0, gamma) * 255.0, 0, 255)
    elif type==1:
        # 线性变换
        color = np.clip(ori * a + b, 0, 255)
    return dst


def read_data(file):

    faces_data = pd.read_csv(file)
    train_set_x = []
    train_set_y = []
    all_set_x = []
    all_set_y = []
    test_set_x = []
    test_set_y = []

    #遍历csv文件内容，并将图片数据按分类保存
    for index in range(len(faces_data)):
        #解析每一行csv文件内容
        emotion_data = faces_data.loc[index][0]
        image_data = faces_data.loc[index][1]
        usage_data = faces_data.loc[index][2]

        emotion_data = keras.utils.to_categorical(emotion_data, 7)
        val = image_data.split(" ")
        pixels = np.array(val, 'float32')

        if usage_data == "Training":
            train_set_y.append(emotion_data)
            train_set_x.append(pixels / 255.0)
        else:
            test_set_x.append(pixels / 255.0)
            test_set_y.append(emotion_data)

        all_set_y.append(emotion_data)
        all_set_x.append(pixels / 255.0)
        all_set_y.append(emotion_data)
        all_set_x.append(brighter(pixels, gamma=0.35) / 255.0)
        all_set_y.append(emotion_data)
        all_set_x.append(brighter(pixels, type=1) / 255.0)

    return train_set_x, train_set_y, test_set_x, test_set_y, all_set_x, all_set_y


# 使用VGG16的改良版
# VGG16:
def my_VGG(in_shape):
    Model = models.Sequential()

    Model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=in_shape))
    Model.add(BatchNormalization())
    Model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    Model.add(BatchNormalization())
    Model.add(MaxPool2D((2, 2), strides=(2, 2), name='block1_pool'))
    Model.add(Dropout(.25))

    # Block 2
    Model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    Model.add(BatchNormalization())
    Model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    Model.add(BatchNormalization())
    Model.add(MaxPool2D((2, 2), strides=(2, 2), name='block2_pool'))
    Model.add(Dropout(.25))

    # Block 3
    Model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    Model.add(BatchNormalization())
    Model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    Model.add(BatchNormalization())
    Model.add(MaxPool2D((2, 2), strides=(2, 2), name='block3_pool'))
    Model.add(Dropout(.25))

    # Block 4
    Model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    Model.add(BatchNormalization())
    Model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    Model.add(BatchNormalization())
    Model.add(MaxPool2D((2, 2), strides=(2, 2), name='block4_pool'))
    Model.add(Dropout(.25))

    # Block 5
    Model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    Model.add(BatchNormalization())
    Model.add(MaxPool2D((2, 2), strides=(2, 2), name='block5_pool'))
    Model.add(Dropout(.25))

    # Classification block
    Model.add(Flatten(name='flatten'))
    Model.add(Dense(2048, activation='relu', name='fc1'))
    Model.add(Dense(2048, activation='relu', name='fc2'))
    Model.add(Dropout(0.5))
    Model.add(Dense(512, activation='relu', name='fc3'))
    Model.add(Dense(7, activation='softmax', name='predictions'))

    return Model


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, data_x, data_y = read_data('./fer2013.csv')

    # 先按照fer2013的标准用测试集和训练集进行一次训练
    # 查看模型的准确率
    train_x = np.array(train_x).reshape(-1, 48, 48, 1)
    train_y = np.array(train_y)
    test_x = np.array(test_x).reshape(-1, 48, 48, 1)
    test_y = np.array(test_y)

    # 所有的数据整合再进行一次训练，最为最终的模型
    data_x = np.array(data_x).reshape(-1, 48, 48, 1)
    data_y = np.array(data_y)

    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        # rotation_range=30,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        horizontal_flip=True)

    model_test = my_VGG(data_x[1].shape)

    adam = Adam()
    model_test.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model_test.summary()

    hist = model_test.fit_generator(data_generator.flow(data_x, data_y, batch_size=32),
                                    epochs=60, verbose=2, validation_data=(test_x, test_y,))

    model_test.save("emotion-final.h5")
