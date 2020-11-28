from PIL import Image
from PIL import ImageFilter
import cv2
import time
import os
import numpy as np
"""代码中是将图片压缩到400X400，可以根据自己的需求修改"""
im = Image.new("RGB", (400, 400), "white") # 生成400X400的白色图片，可以根据自己的需求改变
imndarray = np.array(im)

path = "/home/zxp/download/dataset/original/retry"  # 原图所在文件夹路径
path1 = "/home/zxp/download/dataset/retry"  # 处理完图片的保存路径
filenames = os.listdir(path)

time1 = time.time()
for i in filenames:
    filename = os.path.join(path, i)
    filename1 = os.path.join(path1, i)
    #image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
    img = Image.open(filename, "r")
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #双三次插值
    height, width = image.shape[:2]  #获取原图像的水平方向尺寸和垂直方向尺寸。
    temp = max(height, width)
    multemp = temp/400
    if height > width:
        res = cv2.resize(image, (int(width / multemp), 400), interpolation=cv2.INTER_AREA)
    elif height < width:
        res = cv2.resize(image, (400, int(height / multemp)), interpolation=cv2.INTER_AREA)
    else:
        res = cv2.resize(image, (400, 400), interpolation=cv2.INTER_AREA)

    # 创建滤波器，使用不同的卷积核
    imgE = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    gary2 = imgE.filter(ImageFilter.DETAIL)
    # #图像点运算
    gary3 = gary2.point(lambda i: i*0.9)
    img_convert_ndarray = cv2.cvtColor(np.asarray(gary3), cv2.COLOR_RGB2BGR)
    height1, width1 = img_convert_ndarray.shape[:2]
    temph = int((400 - height1)/2)
    tempw = int((400 - width1)/2)
    a = cv2.copyMakeBorder(img_convert_ndarray, temph, 400-temph-height1, tempw, 400-tempw-width1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.imencode('.jpg', a)[1].tofile(filename1)  # 保存图片
time2 = time.time()
print(u'总共耗时：' + str(time2 - time1) + 's')
