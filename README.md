# Traffic Sign Recognition Project Report

---
The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./output_images/test_img_thumb.png "Thumbnails of trainning Images"
[image2]: ./output_images/class_freq.png "Class frequency count"
[image3]: ./output_images/undistored_perpsective_points.jpg "Source & Destination points for perspective tranform"
[image4]: ./output_images/lines_orig_udist.jpg "Source and Destination point verification"
[image5]: ./output_images/warped_unwarped.jpg "Warp & Unwarped Images"
[image6]: ./output_images/lanes_detection.jpg "Lane detections"
[video1]: ./output_video.mp4 "Video"

## Implementation

This project was implemented within the `Traffic_Sign_Classifier.ipynb` jupyter notebook . The dataset images, where downloaded from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads). Currently, it was possible to download a dataset for trainning, validation and testing. The pickle files can be found under the traffic-sign-data folder. The augmented images can be found under the data folder.

## 1. Loading the Data

The code for this step is contained in section 1 of the IPython notebook `Traffic_Sign_Classifier.ipynb`. For each pickle file I created two arrays one for the features and one for the coresponding labels. Section 1.1 of the IPython notebook includes some statistics for the provided data. The class names are read from `signnames.csv`. In section 1.2 of the jupyter notebook, a random traffic sign for each class is displayed, along with its corresponding class label attached.

![alt text][image1]

A histogram of the distribution of the various samples per class is presented below.

![alt text][image2]

## 1.1 Loading the Data
