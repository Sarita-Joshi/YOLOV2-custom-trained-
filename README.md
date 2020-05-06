## Intro

You can see the original implementation and tutorial for darkflow [here](https://github.com/thtrieu/darkflow)


See demo below or see on [this imgur](demo.gif)

<p align="center"> <img src="demo.gif"/> </p>

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



## Run the model using `flow`

```bash
# Have a look at its options
flow --h
```

First, let's take a closer look at one of a very useful option `--load`

```bash
# 1. Load yolov2-voc.weights
flow --model cfg/yolov2-voc-1c.cfg --load bin/yolov2-voc.weights

# 2. To completely initialize a model, leave the --load option
flow --model cfg/yolov2-voc-1c.cfg

# 3. It is useful to reuse the first identical layers of tiny for `yolo-new`
flow --model cfg/yolov2-voc-1c.cfg --load bin/yolov2-voc.weights
# this will print out which layers are reused, which are initialized
```

All input images from default folder `test_cards/` are flowed through the net and predictions are put in `test_cards/out/`. We can always specify more parameters for such forward passes, such as detection threshold, batch size, images folder, etc.

```bash
# Forward all images in test_cards/ using yolo and 100% GPU usage
flow --imgdir test_cards/ --model cfg/yolov2-voc-1c.cfg --load bin/yolov2-voc.weights --gpu 1.0
```
json output can be generated with descriptions of the pixel location of each bounding box and the pixel location. Each prediction is stored in the `sample_img/out` folder by default. An example json array is shown below.
```bash
# Forward all images in test_cards/ using yolo and JSON output.
flow --imgdir test_cards/ --model cfg/yolov2-voc-1c.cfg --load bin/yolov2-voc.weights --json
```
JSON output:
```json
[{"label": "ace", "confidence": 0.32, "topleft": {"x": 46, "y": 114}, "bottomright": {"x": 208, "y": 227}}, 
{"label": "king", "confidence": 0.26, "topleft": {"x": 303, "y": 182}, "bottomright": {"x": 369, "y": 316}}, 
{"label": "queen", "confidence": 0.49, "topleft": {"x": 182, "y": 351}, "bottomright": {"x": 349, "y": 444}}, 
{"label": "nine", "confidence": 0.27, "topleft": {"x": 48, "y": 0}, "bottomright": {"x": 169, "y": 116}}, 
{"label": "ten", "confidence": 0.39, "topleft": {"x": 62, "y": 304}, "bottomright": {"x": 191, "y": 420}}]
```
 - label: self explanatory
 - confidence: somewhere between 0 and 1 (how confident yolo is about that detection)
 - topleft: pixel coordinate of top left corner of box.
 - bottomright: pixel coordinate of bottom right corner of box.

## Training the model

Training is simple as you only have to add option `--train`. Training set and annotation will be parsed if this is the first time a new configuration is trained. To point to training set and annotations, use option `--dataset` and `--annotation`. A few examples:

```bash
# Initialize yolo from yolov2-voc, then train the net on 100% GPU:
flow --model cfg/yolov2-voc-1c.cfg --load bin/yolov2-voc.weights --train --gpu 1.0

# Completely initialize yolo and train it with ADAM optimizer
flow --model cfg/yolov2-voc-1c.cfg --train --trainer adam
```

During training, the script will occasionally save intermediate results into Tensorflow checkpoints, stored in `ckpt/`. To resume to any checkpoint before performing training/testing, use `--load [checkpoint_num]` option, if `checkpoint_num < 0`, `darkflow` will load the most recent save by parsing `ckpt/checkpoint`.

```bash
# Resume the most recent checkpoint for training
flow --train --model cfg/yolov2-voc-1c.cfg --load -1

# Test with checkpoint at step 1500
flow --model cfg/yolov2-voc-1c.cfg --load 1500

# Fine tuning yolo-tiny from the original one
flow --train --model cfg/yolov2-voc-1c.cfg --load bin/yolov2-voc.weights
```

### Training this model on your own dataset

*The steps below assume we want to use tiny YOLO and our dataset has 3 classes*

1. Create a copy of the configuration file `yolov2-voc.cfg` and rename it according to your preference `yolov2-voc-1c.cfg` (It is crucial that you leave the original `yolov2-voc.cfg` file unchanged, see below for explanation).

2. In `yolov2-voc-1c.cfg`, change classes in the [region] layer (the last layer) to the number of classes you are going to train for. In our case, classes are set to 3.
    
    ```python
    ...

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    bias_match=1
    classes=3
    coords=4
    num=5
    softmax=1
    
    ...
    ```

3. In `yolov2-voc-1c.cfg`, change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and classes are 6 so 5 * (3 + 5) = 40 therefore filters are set to 40.
    
    ```python
    ...

    [convolutional]
    size=1
    stride=1
    pad=1
    filters=40
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
