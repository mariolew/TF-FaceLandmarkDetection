# TF-FaceLandmarkDetection

Face landmark detection using tensorflow
Reproduction of the paper **Deep Convolutional Network Cascade for Facial Point Detection**

## Usage

- git clone https://github.com/mariolew/TF-FaceLandmarkDetection/edit/master
- Prepare data: You should have a text file, each line of the text file should have the format: image_path bbx_left bbx_right bbx_top bbx_bottom landmark1_x landmark1_y ... landmarki_x landmarki_y
- Modify the text file path and the path to store augmented images in **augment.py** and do *python3 augment.py*
- Modify some paths and params in **model_train.py** and do *python3 model_train.py* to train a face alignment model
- Modify some paths and params in **model_eval.py** and do *python3 model_eval.py* to evaluate the trained model

## Note

This repo is based on https://github.com/luoyetx/deep-landmark and https://github.com/pkmital/CADL and is still ongoing.

## Achievements

Level1: Done

Level2: TODO

Level3: TODO

## References

**[1]** [Deep Convolutional Network Cascade for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)

**[2]** [deep-landmark](https://github.com/luoyetx/deep-landmark)

**[3]** [Creative Applications of Deep Learning w/ Tensorflow](https://github.com/pkmital/CADL)

