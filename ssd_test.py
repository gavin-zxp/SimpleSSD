# 
# 测试
#
import os
import sys
from keras import backend as K
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_encoder_decoder.ssd_output_decoder import decode_detections


def start_engine():
    # * 设置图像的高度, 宽度, 和色彩通道数量
    # * 设置正样本类别个数 (不包括背景).
    # * `normalize_coords` 为 True 的时候, 会将所有绝对坐标值转换为相对于图像宽度和高度的值. 这个设置不影响最终的输出.
    img_height = 312  # 图像的高度
    img_width = 416  # 图像的宽度
    n_classes = 3  # 正样本的类别 (不包括背景)
    normalize_coords = True  # 是否使用相对于图像尺寸的相对坐标
    # 设置要加载的模型的路径.
    model_path = 'ssd7_v3_epoch-38_loss-0.9415_val_loss-0.6735.h5'
    # 创建 SSDLoss 对象
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    K.clear_session()  # 从内存中清理曾经加载的模型.
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'compute_loss': ssd_loss.compute_loss})
    # 3. 做预测
    while True:
        filename = input("*" * 50 + "\nPlease enter a image file path:\n")
        # filename = "./data/IMG_20201106_005343.jpg"
        if filename.lower() == "exit":
            print('Program exit! Bye!')
            sys.exit()
        if len(filename.strip()) == 0:
            continue
        if not os.path.isfile(filename):
            print(f"Input '{filename}' is not a file.")
            continue
        print("Image file: ", filename)
        images_numpy = []
        try:
            with Image.open(filename) as image:
                images_numpy.append(np.array(image, dtype=np.uint8))
        except OSError:
            print(f"Input '{filename}' can not be open as an image.")
            continue
        images_numpy = np.array(images_numpy)
        y_pred = model.predict(images_numpy)
        # 4: 解码 `y_pred`
        # 如果我们训练是设置的是 'inference' 或者 'inference_fast' mode, 那么模型的最后一层为 `DecodeDetections` 层,
        # `y_pred` 就无需解码了. 但是我们选择了 'training' mode, 模型的原始输出需要解码. 这就是 `decode_detections()`
        # 这个函数的功能. 这个函数的功能和 `DecodeDetections` 层做的事情一样, 只是使用 Numpy 而不是 TensorFlow 实现.
        # (Nunpy 只能使用CPU, 而不是GPU).
        #
        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.005,
                                           top_k=200,
                                           normalize_coords=normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width)
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("预测值:\n")
        print('   类别   概率   xmin   ymin   xmax   ymax')
        print(y_pred_decoded[0])
        # 5: 在图像上画边界框
        # 最后, 我们可以将预测的边界框画在图像上. 每一个预测的边界框都有类别名称和概率显示在边上.
        plt.figure(figsize=(20, 12))
        plt.imshow(images_numpy[0])
        current_axis = plt.gca()
        colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()  # 设置边界框的颜色
        # classes = ['background', 'bank_card', 'driver_license', 'id_card']  # 类别的名称
        classes = ['背景', '银行卡', '驾照', '身份证']  # 类别的名称
        # 画预测的边界框
        for box in y_pred_decoded[0]:
            xmin = box[-4]
            ymin = box[-3]
            xmax = box[-2]
            ymax = box[-1]
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
        plt.show()


if __name__ == "__main__":
    matplotlib.rc("font", family='KaiTi')
    start_engine()
