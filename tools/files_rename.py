import os
import sys


def rename():
    # pseudo_data
    # path = 'D:/01_DEEPLAENING_FOLDER_by_LIN/DEEPLEARNING/LINTorch/DATA/pseudo_data/images/'
    # path = 'D:/01_DEEPLAENING_FOLDER_by_LIN/DEEPLEARNING/LINTorch/DATA/pseudo_data/labels/'
    path = 'C:/Users/布鲁瓦丝甜甜文/Desktop/常见哺乳动物数据集/'
    # path = 'D:/01_DEEPLAENING_FOLDER_by_LIN/DEEPLEARNING/LINTorch/DATA/labels/'
    name = input("请输入开头名:")
    start_number = input("请输入开始数:")
    file_type = input("请输入后缀名（如 .jpg、.txt等等）:")
    count = 0
    file_list = os.listdir(path)
    # file_list.sort(key=lambda x: int(x[5:-4]))
    print("检查 >>", file_list)
    print("请等候，正在生成以" + name + start_number + file_type + "迭代的文件名...")
    for files in file_list:
        old_dir = os.path.join(path, files)
        if os.path.isdir(old_dir):
            continue
        new_dir = os.path.join(path, name + str(count + int(start_number)) + file_type)
        os.rename(old_dir, new_dir)
        count += 1
    print("一共修改了" + str(count) + "个文件")


rename()
