An Object Detection project based on SSD
=

This is a PyCharm project.
The data set was my certificate took by my phone, and will not be served now.
This model is simple than standard SSD, because my Object Detection task is simple, and it just has 3 classes of target.
Config params in ssd_train.py and run it.

OS: Windows / Linux

Dependency: Tensorflow, Keras, opencv, pillow, matplotlib, scikit-learn ...

Tensorflow and Keras' api is changing, so, open and close the right comment of python code in file keras_laysers/keras_layer_AnchorBoxes.py according to your keras version:
```python
if K.image_dim_ordering() == 'tf':  # keras old version api (works with keras=2.1.5)
if K.image_data_format() == 'channels_last':  # keras new version api (works with keras=2.2.5)
```

I used with tensorflow=1.8.0 and 1.14.0 both works, and with/without gpu both works.
