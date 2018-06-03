import matplotlib.image as mpimg
from sklearn.decomposition import PCA

'''
    《费城》照片处理
'''
# 读取和代码处于同一目录下的图片
image = mpimg.imread('.\dataset\image.jpg')

# 此时图片就已经是一个np.array，可以对它进行任意处理
print(image.shape)  # (2448, 3264, 3)

pca = PCA(n_components='mle')
new_image = pca.fit_transform(image[:, :, 0])
print(new_image.shape)

# # 显示图片
# plt.imshow(new_image)
# # 不显示坐标轴
# plt.axis('off')
# # 激活
# plt.show()

# new_image = functions.pca_reduction(image, 3, 600, 800)

# print(new_image.shape)
# print(new_image)

# # 显示图片
# plt.imshow(image)
# # 不显示坐标轴
# plt.axis('off')
# # 激活
# plt.show()

'''
data = sio.loadmat('./dataset/PaviaU.mat')

paviaU = data['paviaU']

# print(paviaU.shape)

# 显示图片
plt.imshow(paviaU)
# 不显示坐标轴
plt.axis('off')
# 激活
plt.show()
'''
