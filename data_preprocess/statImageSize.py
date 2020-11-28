import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = os.path.join(r'C:\迅雷下载\image')
img_list = os.listdir(path)
all_size = len(img_list)
x = []
y = []
for idx in range(all_size):
    if idx % 100 == 0:
        print('process: %g/%g' % (idx, all_size))
    cur_img = mpimg.imread(os.path.join(path, img_list[idx]))
    x.append(cur_img.shape[1])
    y.append(cur_img.shape[0])
plt.scatter(x, y, marker='+', color='blue', s=10)
plt.xlabel('width')
plt.ylabel('height')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.show()
