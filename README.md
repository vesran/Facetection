# Facetection
Face detection, age, name & gender recognition from streaming, videos, photos...


## Getting started

### Introduction

This is a Computer Vision project based on Deep Learning models to predict carateristics from an image of a person's face. Faces can be detected through several media : 
* Video
* Webcam
* Digital images

<img src="images/friends1.jpg" width="500" title="Example">


### Dependencies

Python 3.8

* Tensorflow
* Numpy
* OpenCV 4.+
* face-recognition (for face embeddings)


### Usage

*TO DO*

## Workflow

### Data

#### Collecting datasets

Models are have been trained on the UTKFace dataset. It provides a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. 

[Link] https://susanqq.github.io/UTKFace/

<img src="images/utk_sample.jpg" width="500" title="Sample">

In order to train the name recognizer, images can be collected through webcam and videos. A new folder is created with the specified name. 

```
datasets/
---- name001
        001.jpg
        002.jpg
        ...
        n.jpg
---- name002
        001.jpg
        002.jpg
        ...
        n.jpg
  ...
```

#### Analysis

More information in ```ÈDA.ipynb```

* **Labels repartition**

<img src="images/ages.jpg" width="600" title="Age repartition">  <img src="images/gender.jpg" width="300" title="Gender repartition"> 

* **Outliers**

Considering outliers as images where we cannot distinguish faces's ROI, 37 files have been detected. Here is a sample :

<img src="images/outliers.jpg" width="800" title="Outliers UTKFace">

### Age recognition

### Gender recognition

### Name recognition

## Acknownledgement
