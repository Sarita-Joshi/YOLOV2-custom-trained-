## Intro

You can see the original implementation and tutorial for darkflow[here](https://github.com/thtrieu/darkflow)


See demo below or see on [this imgur](video.avi)

<p align="center"> <img src="video.avi"/> </p>

## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3.


### Getting started

You can choose _one_ of the following three ways to get started with darkflow.

1. Just build the Cython extensions in place. NOTE: If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.
    ```
    python3 setup.py build_ext --inplace
    ```

2. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

3. Install with pip globally
    ```
    pip install .
    ```

## Update

**Android demo on Tensorflow's** [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowYoloDetector.java)

**I am looking for help:**
 - `help wanted` labels in issue track

## Parsing the annotations

Skip this if you are not training or fine-tuning anything (you simply want to forward flow a trained net)

For example, if you want to work with only 3 classes `tvmonitor`, `person`, `pottedplant`; edit `labels.txt` as follows

```
tvmonitor
person
pottedplant
```

And that's it. `darkflow` will take care of the rest. You can also set darkflow to load from a custom labels file with the `--labels` flag (i.e. `--labels myOtherLabelsFile.txt`). This can be helpful when working with multiple models with different sets of output labels. When this flag is not set, darkflow will load from `labels.txt` by default (unless you are using one of the recognized `.cfg` files designed for the COCO or VOC dataset - then the labels file will be ignored and the COCO or VOC labels will be loaded).

## Design the net

Skip this if you are working with one of the original configurations since they are already there. Otherwise, see the following example:

```python
...

[convolutional]
batch_normalize = 1
size = 3
stride = 1
pad = 1
activation = leaky

[maxpool]

[connected]
output = 4096
activation = linear

...
```

## Flowing the graph using `flow`

```bash
# Have a look at its options
flow --h
```

First, let's take a closer look at one of a very useful option `--load`

```bash
# 1. Load tiny-yolo.weights
flow --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights

# 2. To completely initialize a model, leave the --load option
flow --model cfg/yolo-new.cfg

# 3. It is useful to reuse the first identical layers of tiny for `yolo-new`
flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights
# this will print out which layers are reused, which are initialized
```

All input images from default folder `sample_img/` are flowed through the net and predictions are put in `sample_img/out/`. We can always specify more parameters for such forward passes, such as detection threshold, batch size, images folder, etc.

```bash
# Forward all images in sample_img/ using tiny yolo and 100% GPU usage
flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --gpu 1.0
```
json output can be generated with descriptions of the pixel location of each bounding box and the pixel location. Each prediction is stored in the `sample_img/out` folder by default. An example json array is shown below.
```bash
# Forward all images in sample_img/ using tiny yolo and JSON output.
flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --json
```
JSON output:
```json
[{"label":"person", "confidence": 0.56, "topleft": {"x": 184, "y": 101}, "bottomright": {"x": 274, "y": 382}},
{"label": "dog", "confidence": 0.32, "topleft": {"x": 71, "y": 263}, "bottomright": {"x": 193, "y": 353}},
{"label": "horse", "confidence": 0.76, "topleft": {"x": 412, "y": 109}, "bottomright": {"x": 592,"y": 337}}]
```
 - label: self explanatory
 - confidence: somewhere between 0 and 1 (how confident yolo is about that detection)
 - topleft: pixel coordinate of top left corner of box.
 - bottomright: pixel coordinate of bottom right corner of box.

## Training new model

Training is simple as you only have to add option `--train`. Training set and annotation will be parsed if this is the first time a new configuration is trained. To point to training set and annotations, use option `--dataset` and `--annotation`. A few examples:

```bash
# Initialize yolo-new from yolov2-voc, then train the net on 100% GPU:
flow --model cfg/yolov2-voc.cfg --load bin/yolov2-voc.weights --train --gpu 1.0

# Completely initialize yolo-new and train it with ADAM optimizer
flow --model cfg/yolov2-voc.cfg --train --trainer adam
```

During training, the script will occasionally save intermediate results into Tensorflow checkpoints, stored in `ckpt/`. To resume to any checkpoint before performing training/testing, use `--load [checkpoint_num]` option, if `checkpoint_num < 0`, `darkflow` will load the most recent save by parsing `ckpt/checkpoint`.

```bash
# Resume the most recent checkpoint for training
flow --train --model cfg/yolov2-voc.cfg --load -1

# Test with checkpoint at step 1500
flow --model cfg/yolov2-voc.cfg --load 1500

# Fine tuning yolo-tiny from the original one
flow --train --model cfg/yolov2-voc.cfg --load bin/yolov2-voc.weights
```

### Training on your own dataset

*The steps below assume we want to use tiny YOLO and our dataset has 3 classes*

1. Create a copy of the configuration file `yolov2-voc.cfg` and rename it according to your preference `yolov2-voc-1c.cfg` (It is crucial that you leave the original `yolov2-voc.cfg` file unchanged, see below for explanation).

2. In `yolov2-voc-1c.cfg`, change classes in the [region] layer (the last layer) to the number of classes you are going to train for. In our case, classes are set to 3.
    
    ```python
    ...

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    bias_match=1
    classes=6
    coords=4
    num=5
    softmax=1
    
    ...
    ```

3. In `yolov2-voc-1c.cfg`, change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and classes are 6 so 5 * (6 + 5) = 55 therefore filters are set to 55.
    
    ```python
    ...

    [convolutional]
    size=1
    stride=1
    pad=1
    filters=55
    activation=linear

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    
    ...
    ```

4. Change `labels.txt` to include the label(s) you want to train on (number of labels should be the same as the number of classes you set in `tiny-yolo-voc-3c.cfg` file). In our case, `labels.txt` will contain 3 labels.

    ```
    label1
    label2
    label3
    ```
5. Reference the `yolov2-voc-1c.cfg` model when you train.

    `flow --model cfg/yolov2-voc-1c.cfg --load bin/yolov2-voc.weights --train --annotation train/annotations --dataset train/images`



## Save the built graph to a protobuf file (`.pb`)

```bash
## Saving the lastest checkpoint to protobuf file
flow --model cfg/yolov2-voc-1c.cfg --load -1 --savepb

## Saving graph and weights to protobuf file
flow --model cfg/yolov2-voc-1c.cfg --load bin/yolov2-voc.weights --savepb
```


That's all.
