import cv2
import random
from scipy.io import *
import pandas as pd
import numpy as np
import tensorflow as tf

pred = pd.read_csv('../dataset/SVM_pred.csv', header=None)
preds = pred.values

print(pred.shape)  # (610, 340)


# 返回一个像素坐标的上下左右四个坐标
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
    if i_up < 0 or i_down > 609 or j_left < 0 or j_right > 339:
        return False, False, False, False
    else:
        return [i_up, j_up], [i_down, j_down], [i_left, j_left], [i_right, j_right]


# 记录联通中心位置和对应联通区域大小的字典
union = dict()

# 记录当前连通的错误分类像素区域坐标
union_set = set()

# 记录当前连通的错误分类像素区域的外围区域坐标
unionbound_set = set()

# 生成标记是否处理过像素的备用数据组，用正常坐标值记录
mark_list = []
for i in range(207400):
    mark_list.append(i)


# 连通区域递归函数
def mani(i, j):
    # 绝对坐标换算正常坐标值
    loc = i * 340 + j
    # 排查是否属于其他联通图处理过，就跳过
    if loc not in mark_list:
        return
    # 移除当前中心像素
    mark_list.remove(loc)
    # 先确定不是0背景类
    if pred[i][j] == 0:
        return
    # 获取当前元素上下左右四个绝对坐标
    up, down, left, right = return_4loc(i, j)
    # 在当前像素处于画面边缘时，跳过
    if not up:
        return
    # up绝对坐标
    i_up = up[0]
    j_up = up[1]
    # down绝对坐标
    i_down = down[0]
    j_down = down[1]
    # left绝对坐标
    i_left = left[0]
    j_left = left[1]
    # right绝对坐标
    i_right = right[0]
    j_right = right[1]
    # 统计上下左右四个坐标
    up_loc = i_up * 340 + j_up
    down_loc = i_down * 340 + j_down
    left_loc = i_left * 340 + j_left
    right_loc = i_right * 340 + j_right
    # 如果当前像素，对应的上下左右四个像素有一个不在gt标记范围内，就暂时跳过，这说明该像素在边缘
    if pred[i_up][j_up] == 0 or pred[i_down][j_down] == 0 or pred[i_left][j_left] == 0 or pred[i_right][
        j_right] == 0:
        return
    # 将当前像素上下左右四个坐标放入包围边缘集合
    if up_loc not in union_set:
        unionbound_set.add(up_loc)
    if down_loc not in union_set:
        unionbound_set.add(down_loc)
    if left_loc not in union_set:
        unionbound_set.add(left_loc)
    if right_loc not in union_set:
        unionbound_set.add(right_loc)
    # 判断上下左右是与当前中心像素被判定为同一类别
    if pred[i_up][j_up] == pred[i][j]:
        # 判定为同一类别，则将其添加入连通区域集合
        union_set.add(up_loc)
        # 将其从边缘区域集合中移除
        unionbound_set.remove(up_loc)
        # 递归处理当前连通的新像素
        mani(i_up, j_up)
    if pred[i_down][j_down] == pred[i][j]:
        union_set.add(down_loc)
        unionbound_set.remove(down_loc)
        mani(i_down, j_down)
    if pred[i_left][j_left] == pred[i][j]:
        union_set.add(left_loc)
        unionbound_set.remove(left_loc)
        mani(i_left, j_left)
    if pred[i_right][j_right] == pred[i][j]:
        union_set.add(right_loc)
        unionbound_set.remove(right_loc)
        mani(i_right, j_right)

    union[loc] = len(union_set)


for i in range(610):
    for j in range(340):
        mani(i, j)

print(union)
