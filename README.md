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
[image3]: ./output_images/lenet_graph.png "Network graph"
[image4]: ./output_images/lines_orig_udist.jpg "Source and Destination point verification"
[image5]: ./output_images/warped_unwarped.jpg "Warp & Unwarped Images"
[image6]: ./output_images/lanes_detection.jpg "Lane detections"
[video1]: ./output_video.mp4 "Video"

## Implementation

This project was implemented within the `Traffic_Sign_Classifier.ipynb` jupyter notebook . The dataset images, where downloaded from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads). Currently, it was possible to download a dataset for trainning, validation and testing. The pickle files can be found under the traffic-sign-data folder. The augmented images can be found under the data folder.

## 1. Loading the Data

The code for this step is contained in section 1 of the IPython notebook `Traffic_Sign_Classifier.ipynb`. For each pickle file I created two arrays one for the features and one for the coresponding labels. Section 1.1 of the IPython notebook includes some statistics for the provided data. The class names are read from the file `signnames.csv`. In section 1.2 of the jupyter notebook, a random traffic sign for each class is displayed, along with its corresponding class label attached.

![alt text][image1]

A histogram of the distribution of the various samples per class is presented below.

![alt text][image2]

## 2 Model Architecture Design and Test

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson was used as a starting point with minor modifications. The number of classes was changed from 10 to 43. Additional `scope` labels where added in order to use Tensor Board for better visualization of the network and being able to better visualize the effect of the various parameter changes have on the trainned model. The graph of the implemented network can be seen in the figure below.

![alt text][image3]

### 2.1 Data pre-processing

The first pre-processing technique used, was normalization. However, there was only a marginal improvement in accuracy. I then converted the image data sets to grayscale. The implementation can be seen within method `preprocess_dataset()` found in section 2.1 of the jupyter notebook. This improved the accuracy (0.90), however performing an adaptive histogram equalization on the entire dataset increased the accuracy even more and gave more concistent results. This is implemented within method `normalize_images()`. Adaptive histogram equalization effectively reduces the effect that bad light conditions (or capture settings of the camera) have on the quality of the image, revealing more details. I also tried to merge these two methods but the result was actually worse. Eventually, only the later method was used. The implementation for both methods, can be found in section 2.1 of the jupyter notebook. The normalized images were saved on a pickle file. The code checks for the existence of this file in order to avoid re-computing the histogram equalization on subsequent executions of the notebook, unless the corresponding file is removed.
In addition to that, the data set was augmented by additing rotated and shifted images. Initially, the implementation was based on scikit, but it was slow and lengthy. I replaced this code with the `ImageDataGenerator()` method available through the Keras library. I was also able to add zoomed images also. The trainning dataset was saved into a pickle file `augmented_data.p` in the output_images folder. A total of 215000 images were used for trainning. 
