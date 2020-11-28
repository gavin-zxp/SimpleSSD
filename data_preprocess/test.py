import numpy as np
# from keras_preprocessing import image
import cv2 as cv
from PIL import Image, ImageOps
from os import path
import glob

'''
# 自己定义的一个（4,4,3）的numpy数组
img_num = np.array([[[10, 30, 60], [100, 120, 150], [77, 99, 130], [200, 30, 59]],
                    [[40, 10, 160], [150, 120, 150], [77, 99, 130], [100, 30, 59]],
                    [[20, 90, 210], [100, 220, 150], [37, 199, 230], [210, 90, 99]],
                    [[100, 40, 40], [200, 50, 20], [157, 9, 140], [50, 230, 119]]])

print(img_num)
print('==============================================')
image.save_img("C:/Users/admin/Documents/data/test/2006.jpg", img_num, scale=True)  # scale默认为true
image.save_img("C:/Users/admin/Documents/data/test/2007.jpg", img_num, scale=False)  # scale设置为false

img_06 = image.load_img("C:/Users/admin/Documents/data/test/2006.jpg")  # img_01 是一个PIL.Image类的实例对象
img_07 = image.load_img("C:/Users/admin/Documents/data/test/2007.jpg")
a_06 = image.img_to_array(img_06)
a_07 = image.img_to_array(img_07)

print(a_06)
print('==============================================')
print(a_07)
'''

# keras preprocessing
'''
img_05 = image.load_img("C:/Users/admin/Documents/data/test/IMG_20201105_215021.jpg")  # 读取图片，转化成PIL.Image对象
num_05 = image.img_to_array(img_05)  # 将PIL.Image对象转化成numpy数组
# 像素值不变
# num_05_ = image.random_zoom(num_05, (10, 10), row_axis=0, col_axis=1, channel_axis=2)
num_05_ = image.random_zoom(num_05, zoom_range=(10, 10), row_axis=0, col_axis=1, channel_axis=2)
image.save_img("C:/Users/admin/Documents/data/test/37.jpg", num_05_)  # 保存旋转之后的图片
'''

# cv2
img = cv.imread("C:/Users/admin/Documents/data/test/ILSVRC2012_val_00000048.jpg")
# print(img.shape)
x, y = img.shape[0:2]
# 显示原图
# cv.imshow('OriginalPicture', img)
# 缩放到原来的十分之一，输出尺寸格式为（宽，高）
# img_test1 = cv.resize(img, (int(y / 10), int(x / 10)), interpolation=cv.INTER_AREA)
# 放大
img_test1 = cv.resize(img, (int(y * 10), int(x * 10)), interpolation=cv.INTER_CUBIC)
cv.imwrite("C:/Users/admin/Documents/data/test/45.jpg", img_test1)


# PIL
def resize_image(filein, fileout, width=600, height=500):
    temp_img = Image.open(filein)
    out = temp_img.resize((width, height), Image.ANTIALIAS)  # resize image with high-quality
    out.save(fileout)


# resize_image("C:/Users/admin/Documents/data/test/IMG_20201105_215021.jpg", "C:/Users/admin/Documents/data/test/47.jpg", 416, 312, 'jpg')
# 批量转换
for jpg_file in glob.glob("E:/test/picture/12/*.jpg"):
    resize_image(jpg_file, path.join("E:/test/picture/111/", path.basename(jpg_file)), 600, 500)


# 只能保持宽高比例缩小，暂时不需要
image = Image.open("C:/Users/admin/Documents/data/test/IMG_20201105_215021.jpg")
fmt = image.format
print(f"format: {fmt}")
# image = ImageOps.mirror(image)  # 水平翻转
new_size = (241, 241)
image.thumbnail(new_size, Image.ANTIALIAS)  # 按照比例缩小，不能放大
image.save("C:/Users/admin/Documents/data/test/48.jpg", fmt)


