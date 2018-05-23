import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import scipy.io as sio
import numpy as np

'''
    PCA压缩：（通道数不变）
    原始图片矩阵、原始图片通道数、目标压缩高度，目标压缩宽度
'''


def pca_reduction(image, channel, height, width):
    new_image = np.zeros((height, width, channel))
    for i in range(channel):
        # 读取当前通道
        image_channel = image[:, :, i]
        # 横向压缩
        pca = PCA(n_components=width)
        image_channel = pca.fit_transform(image_channel)
        # 纵向压缩
        image_channel = image_channel.T
        pca = PCA(n_components=height)
        image_channel = pca.fit_transform(image_channel)
        # 重新合成
        image_channel = image_channel.T
        new_image[:, :, i] = image_channel
    return new_image
