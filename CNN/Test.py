import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, AveragePooling2D, Flatten, Reshape
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization


# from lrn import LRN  # custom LRN implementation in Keras

def convertToNeighbours(a, window, step_size):
    shape = a.shape[:-1] + (window, window) + a.shape[-1:]
    print('Sahpe is ', shape)
    # strides = a.strides + (a.strides[-1] * step_size,)
    strides = (1, 1, 5, 5, 200)
    print('Strieds: ', strides)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# plot ground truth as an image
def plotData(gt):
    # plt.plot(gt)
    plt.title('Indiana Pines')
    plt.imshow(gt)
    plt.set_cmap('nipy_spectral')
    plt.show()


# Load data from .mat files and return test and train split
def loadData(dataSetPath, gtPath):
    dataSet = sio.loadmat(dataSetPath)['indian_pines_corrected']
    # print(dataSet.shape[:-1])
    Y = sio.loadmat(gtPath)['indian_pines_gt']
    # print(gt.shape)
    X = convertToNeighbours(dataSet, 5, 1)
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)
    # print(np.unique(Y))
    plotData(Y)
    return X, Y


dataX, dataY = loadData('../dataset/Indian_pines_corrected.mat', '../dataset/Indian_pines_gt.mat')

dataX = dataX.reshape([-1, 5, 5, 200])
dataY = dataY.reshape([145 * 145])
print('uniq', np.unique(dataY))
print(dataY)
# dataY = to_categorical(dataY, 17)
# print('New X shape: ', dataX.shape)
# print('New Y shape: ', dataY.shape)

(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, test_size=0.6, random_state=42)

print('Train X and Y: ', trainX.shape, trainY.shape)
print('Test X and Y: ', testX.shape, testY.shape)

augm = ImageDataGenerator(rotation_range=270.0, horizontal_flip=True)
trainGen = augm.flow(trainX, trainY, batch_size=16)
testGen = augm.flow(testX, testY, batch_size=16)

trainSteps = trainX.shape[0] // 16
testSteps = testX.shape[0] // 16
# augm.flow(dataX, dataY)

model = Sequential()
model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', input_shape=(5, 5, 200), activation='relu'))
model.add(BatchNormalization())
# model.add(LRN(alpha=0.0001, beta=0.75, k=3))
model.add(Dropout(0.6))
model.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
model.add(Dropout(0.6))
# model.add(LRN(alpha=0.0001, beta=0.75, n=3))
model.add(BatchNormalization())
model.add(Conv2D(17, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
# model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(AveragePooling2D(strides=(1, 1), pool_size=(5, 5), data_format='channels_last'))
model.add(Reshape((17,)))
model.summary()
opt = SGD(lr=0.001, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
