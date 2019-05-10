import cv2
import random
from scipy.io import *
import pandas as pd
import numpy as np
import tensorflow as tf

pred = pd.read_csv('../dataset/test.csv', header=None)
preds = pred.values

print(pred.shape)  # (6, 5)

pred = np.transpose(pred)


# 给定一个二维（x，y）坐标，返回一个像素坐标的上下左右四个（x，y）坐标
def return_4loc(i, j):
    # up
    i_up = i - 1
    j_up = j
    # down
    i_down = i + 1
    j_down = j
    # left
    i_left = i
    j_left = j - 1
    # right
    i_right = i
    j_right = j + 1
    # 超出边界的像素返回四个FALSE
    if i_up < 0 or i_down > 5 or j_left < 0 or j_right > 4:
        return False, False, False, False
    else:
        # 否则返回四个坐标
        # print([i_up, j_up], [i_down, j_down], [i_left, j_left], [i_right, j_right])
        return [i_up, j_up], [i_down, j_down], [i_left, j_left], [i_right, j_right]


# 记录当前连通区域的一维坐标集合
unionloc_set = set()
# 记录当前连通区域边界像素的一维坐标集合
unionbound_set = set()

# 记录当前中心位置和对应联通区域大小的字典
union_size = dict()

#
flag = True


# 连通区域递归函数，传入中心目标像素的二维坐标（x，y）
def mani(i, j):
    # 记录当前中心像素的判定结果
    current_pred = pred[i][j]
    # 二维坐标换算一维坐标值备用
    loc = i * 5 + j

    # 获取当前元素上下左右四个二维坐标
    up, down, left, right = return_4loc(i, j)
    # 在当前像素处于画面边缘时，跳过
    if not up or not down or not left or not right:
        global flag
        flag = False
        return
    # 将获取到的上下左右四个二维坐标详细记录
    # up坐标
    i_up = up[0]
    j_up = up[1]
    # down坐标
    i_down = down[0]
    j_down = down[1]
    # left坐标
    i_left = left[0]
    j_left = left[1]
    # right坐标
    i_right = right[0]
    j_right = right[1]
    # 统计上下左右四个的一维坐标
    up_loc = i_up * 5 + j_up
    down_loc = i_down * 5 + j_down
    left_loc = i_left * 5 + j_left
    right_loc = i_right * 5 + j_right
    # 如果当前像素，对应的上下左右四个像素有一个不在gt标记范围内，就暂时跳过，这说明该像素在边缘
    if pred[i_up][j_up] == 0 or pred[i_down][j_down] == 0 or pred[i_left][j_left] == 0 or pred[i_right][
        j_right] == 0:
        flag = False
        return
    # 否则既然属于标记范畴，开始判定连通问题

    # 将当前中心元素率先放入连通集合
    unionloc_set.add(loc)

    # 再将中心区域上下左右四个像素放入边界集合
    if up_loc not in unionloc_set:
        unionbound_set.add(up_loc)
    if down_loc not in unionloc_set:
        unionbound_set.add(down_loc)
    if left_loc not in unionloc_set:
        unionbound_set.add(left_loc)
    if right_loc not in unionloc_set:
        unionbound_set.add(right_loc)

    for pix in list(unionbound_set):
        i_pix = pix // 5
        j_pix = pix % 5
        if pred[i_pix][j_pix] == current_pred:
            unionloc_set.add(pix)
            try:
                unionbound_set.remove(pix)
            except BaseException:
                print('')

            mani(i_pix, j_pix)

    # 设定阈值将小面积联通区域找出并记录
    # if len(unionloc_set) < 20:
    #     union_size[(i, j)] = len(unionloc_set)
    # print(union_size)


for i in range(6):
    for j in range(5):
        print((i, j))
        # 记录当前连通区域的一维坐标集合
        unionloc_set = set()
        # 记录当前连通区域边界像素的一维坐标集合
        unionbound_set = set()
        # 确定联通区域是否触及边界
        flag = True
        if pred[i][j] != 0:
            mani(i, j)
            if len(unionloc_set) < 20 and len(unionloc_set) != 0 and flag:
                union_size[(i, j)] = len(unionloc_set)
                for pix in unionloc_set:
                    i_pix = pix // 5
                    j_pix = pix % 5
                    pred[i_pix][j_pix] = 0

            # unionloc_set.clear()
            # unionbound_set.clear()

print(union_size)
