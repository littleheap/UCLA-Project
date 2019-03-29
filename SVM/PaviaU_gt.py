from scipy.io import loadmat
import cv2

'''
    将验证集paviaU_gt中10中类别（9中分类+1个无类别背景）用不同颜色标记出来并以RGB格式图片显示
'''
# 读取标记图片paviaU_gt
output_image = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']

c2 = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']

c3 = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']

# 对每一种类别用一种颜色在三个通道上标记
for i in range(610):
    for j in range(340):
        if (output_image[i][j] == 0):
            c1[i][j] = 255
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 1):
            c1[i][j] = 20
            c2[i][j] = 104
            c3[i][j] = 82
        if (output_image[i][j] == 2):
            c1[i][j] = 40
            c2[i][j] = 200
            c3[i][j] = 160
        if (output_image[i][j] == 3):
            c1[i][j] = 60
            c2[i][j] = 240
            c3[i][j] = 111
        if (output_image[i][j] == 4):
            c1[i][j] = 80
            c2[i][j] = 77
            c3[i][j] = 190
        if (output_image[i][j] == 5):
            c1[i][j] = 14
            c2[i][j] = 80
            c3[i][j] = 90
        if (output_image[i][j] == 6):
            c1[i][j] = 120
            c2[i][j] = 60
            c3[i][j] = 150
        if (output_image[i][j] == 7):
            c1[i][j] = 140
            c2[i][j] = 200
            c3[i][j] = 255
        if (output_image[i][j] == 8):
            c1[i][j] = 160
            c2[i][j] = 5
            c3[i][j] = 100
        if (output_image[i][j] == 9):
            c1[i][j] = 180
            c2[i][j] = 180
            c3[i][j] = 255

# 合并三个通道，组成三通道RGB图片
multi_merged = cv2.merge([c1, c2, c3])

# 显示图片
cv2.imshow("output", multi_merged)

# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

# 存储图片
cv2.imwrite('../images/paviaU.png', multi_merged)

# 单颜色（黄色）显示标记图片

# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']

c2 = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']

c3 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']

for i in range(610):
    for j in range(340):
        if (output_image[i][j] == 0):
            c1[i][j] = 255
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 1):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 2):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 3):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 4):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 5):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 6):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 7):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 8):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 9):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255

# 合并三个通道，组成三通道RGB图片
single_merged = cv2.merge([c1, c2, c3])

# 显示图片
cv2.imshow("output", single_merged)

# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

# 存储图片
cv2.imwrite('../images/paviaU.png', single_merged)
