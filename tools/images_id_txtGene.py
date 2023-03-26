# 请确保文件夹内文件命名仅有数字
import os

trainPath = 'D:/train/'
txtPath = 'D:/'

f = open(txtPath + 'images_id.txt', 'w+')  # 没有则创建txt文件，与代码文件同目录
fileList = os.listdir(trainPath)
# fileList.sort(key=lambda x: int(x[4:-4]))  # 5：i m a g e共5个字符
print('次序预览 >> ', fileList)
for file in fileList:
    f.write(trainPath + str(file) + '\n')  # 保留文件后缀

f.close()
