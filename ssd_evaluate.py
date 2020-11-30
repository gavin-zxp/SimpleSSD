# 
# 训练一个 SSD 网络用于识别车载摄像头捕捉的图像中的目标
#
import os
import sys
from keras.models import load_model
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator

# ## 1. 设置模型参数
#
# 这些参数同时需要在 `build_model()` 和 `SSDInputEncoder` 的构造函数中使用
# * 设置图像的高度, 宽度
# * 设置正样本类别个数 (不包括背景).
# * 网络输出原始预测值. 当值为 'inference' 或者 'inference_fast' 时, 原始的输出被转为了绝对坐标, 使用了概率筛选, 使用
# * 了 non-maximum suppression, 以及 top-k 筛选. 'inference' 使用了原始Caffe实现的算法, 'inference_fast' 使用了
# * 更快, 但是相对不太精确的算法 
# * `normalize_coords` 为 True 的时候, 会将所有绝对坐标值转换为相对于图像宽度和高度的值. 这个设置不影响最终的输出.

img_height = 312  # 图像的高度
img_width = 416  # 图像的宽度
classes = ['总体', '银行卡', '驾照', '身份证']  # 类别的名称
n_classes = 3  # 正样本的类别 (不包括背景)
normalize_coords = True  # 是否使用相对于图像尺寸的相对坐标

# 加载一个训练好的模型
model_path = 'ssd7_v3_epoch-38_loss-0.9415_val_loss-0.6735.h5'
# 创建 SSDLoss 对象
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
K.clear_session()  # 从内存中清理曾经加载的模型.
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': ssd_loss.compute_loss})

eval_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# Images
images_dir = './data/'
out_dir = './predict_error'
# Ground truth
eval_labels_filename = './data/labels_eval.csv'
eval_dataset.parse_csv(images_dir=images_dir,
                       labels_filename=eval_labels_filename,
                       input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                       include_classes='all')
# 得到训练和validation数据集的数据的量.
eval_dataset_size = eval_dataset.get_dataset_size()
print("evaluate集的图像数量\t{:>6}".format(eval_dataset_size))
predict_generator = eval_dataset.generate(batch_size=eval_dataset_size,
                                          shuffle=True,
                                          transformations=[],
                                          label_encoder=None,
                                          returns={'processed_images',
                                                   'processed_labels',
                                                   'filenames'},
                                          keep_images_without_gt=False)
batch_images, batch_labels, batch_filenames = next(predict_generator)
# 3: 作预测
y_pred = model.predict(batch_images)
# 4: 解码 `y_pred`
# 如果我们训练是设置的是 'inference' 或者 'inference_fast' mode, 那么模型的最后一层为 `DecodeDetections` 层,
# `y_pred` 就无需解码了. 但是我们选择了 'training' mode, 模型的原始输出需要解码. 这就是 `decode_detections()`
# 这个函数的功能. 这个函数的功能和 `DecodeDetections` 层做的事情一样, 只是使用 Numpy 而不是 TensorFlow 实现.
# (Nunpy 只能使用CPU, 而不是GPU).
y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)


def calc_pr(y_predict, image_list, labels_list, file_name_list):
    error_file_idx = set()
    # tp, fp, fn, p, r
    #  0,  1,  2, 3, 4
    stat_np = np.zeros((n_classes + 1, 5), dtype=np.float)
    for idx in range(len(file_name_list)):
        if len(y_predict[idx]) == 0:
            pred_labels = set()
        else:
            pred_labels = set(y_predict[idx][..., 0].astype(np.int32).tolist())
        gt_labels = set(labels_list[idx][..., 0].tolist())
        tp = gt_labels & pred_labels
        fp = pred_labels - gt_labels
        fn = gt_labels - pred_labels
        stat_np[0, 0] += len(tp)
        stat_np[0, 1] += len(fp)
        stat_np[0, 2] += len(fn)
        if len(fp | fn) > 0:
            error_file_idx.add(idx)
            plt.figure(figsize=(20, 12))
            plt.imshow(image_list[idx])
            current_axis = plt.gca()
            label_names = [classes[int(lb)] for lb in pred_labels]
            current_axis.text(0, 0, ', '.join(label_names), size='x-large', color='white',
                              bbox={'facecolor': 'green', 'alpha': 1.0})
            plt.savefig(os.path.join(out_dir, os.path.basename(file_name_list[idx])))
            plt.close()
        for label_idx in tp:
            stat_np[int(label_idx), 0] += 1
        for label_idx in fp:
            stat_np[int(label_idx), 1] += 1
        for label_idx in fn:
            stat_np[int(label_idx), 2] += 1
    stat_np[..., 3] = stat_np[..., 0] / (stat_np[..., 0] + stat_np[..., 1])
    stat_np[..., 4] = stat_np[..., 0] / (stat_np[..., 0] + stat_np[..., 2])
    # stat_np = np.insert(stat_np, 0, np.arange(n_classes + 1), axis=1)
    with open('./evaluate_result.txt', 'a', encoding='utf-8') as f:
        f.write('=' * 30 + '\n')
        f.write('类别 [TP FP FN P R]\n')
        for idx in range(n_classes + 1):
            f.write(classes[idx] + '  ' + str(stat_np[idx]) + '\n')
        f.write('*' * 7 + 'error file names' + '*' * 7 + '\n')
        for idx in error_file_idx:
            f.write(file_name_list[idx] + '\n')


matplotlib.rc("font", family='KaiTi')
calc_pr(y_pred_decoded, batch_images, batch_labels, batch_filenames)
print("Finish evaluate!")
