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
[image4]: ./output_images/test_signs.png "Belgian traffic signs for testing"
[image5]: ./output_images/classified_test_signs.png "Classified belgian signs"
[image6]: ./output_images/classified_test_signs_french.png "Classified french signs"

## Implementation

This project was implemented within the `Traffic_Sign_Classifier.ipynb` jupyter notebook . The dataset images, where downloaded from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads). Currently, it was possible to download a dataset for trainning, validation and testing. The pickle files can be found under the traffic-sign-data folder. The augmented images can be found under the data folder.

## 1. Loading the Data

The code for this step is contained in section 1 of the IPython notebook `Traffic_Sign_Classifier.ipynb`. For each pickle file I created two arrays one for the features and one for the coresponding labels. Section 1.1 of the IPython notebook includes some statistics for the provided data. The class names are read from the file `signnames.csv`. In section 1.2 of the jupyter notebook, a random traffic sign for each class is displayed, along with its corresponding class label attached.

![alt text][image1]

A histogram of the distribution of the various samples per class is presented below.

![alt text][image2]

## 2 Model Architecture Design and Test

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson was used as a starting point with minor modifications. The number of classes was changed from 10 to 43. Additional `scope` labels where added in order to utilize Tensor Board for better visualization of the network performace. In addition to that, tensor board allows for better visualization of the effects that various hyper parameter changes have on the trainned model. The graph of the implemented network can be seen in the figure below.

![alt text][image3]

The initial validation accuracy, of the LeNet-5 model was about 0.89, using the original dataset. Augmenting and normalizing the data increased the validation accuracy to 0.93.
The optimizer was also changed from SGD to Adam and various learning rates were tested. The optimal rate was found to be 0.0015 (three rates were tested: 0.0015 0.0020 0.0025). With the Adam optimizer and this rate we were able to achieve an accuracy of 0.968 on the validation data and 0.951 on the test data. Higher learning rates converged faster but the validation accuracy was lower. I also tried exponential learning decay (code is commented out on the jupyter notebook) but the result was marginally worse and therefore it was not used.
I also tried L2 regularization (before modifying the code to support tensor board) but this did not improve accuracy either (code is commented out on the jupyter notebook).
Batch size was set to 128. 256 was also tested but the resulting accuracy was unaffected. The model trainned for 10 epochs, as this seems to be the limit where overfitting is observed.

### 2.1 Data pre-processing

The first pre-processing technique used, was normalization. However, there was only a marginal improvement in accuracy. I then converted the image data sets to grayscale. The implementation can be seen within method `preprocess_dataset()` found in section 2.1 of the jupyter notebook. This improved the accuracy (0.90), however performing an adaptive histogram equalization on the entire dataset increased the accuracy even more and gave more concistent results. This is implemented within method `normalize_images()`. Adaptive histogram equalization effectively reduces the effect that bad light conditions (or capture settings of the camera) have on the quality of the image, revealing more details. I also tried to merge these two methods but the result was actually worse. Eventually, only the later method was used. The implementation for both methods, can be found in section 2.1 of the jupyter notebook. The normalized images were saved on a pickle file. The code checks for the existence of this file in order to avoid re-computing the histogram equalization on subsequent executions of the notebook, unless the corresponding file is removed.
In addition to that, the data set was augmented by additing rotated and shifted images. Initially, the implementation was based on scikit, but it was slow and lengthy. I replaced this code with the `ImageDataGenerator()` method available through the Keras library. I was also able to add zoomed images also. The trainning dataset was saved into a pickle file `augmented_data.p` in the output_images folder. A total of 215000 images were used for trainning.

### 3 Testing the model on new images

Ten images from the Belgian traffic sign set were downloaded, as they are pretty similar to the German traffic signs. These are reference images i.e. not taken a document. The class names were read from signnames.csv as it contains mappings fom the class id (integer) to the actual sign name. The images can be seen on the figure below:

![alt text][image4]

Since these were reference images the classification was perfect as can be seen from the figure below:

![alt text][image5]

A second set of images, taken from actual pictures was also used to test the classification accuracy when real world image are used. These traffic signs were from a French traffic sign set. Some of the signs were missclassified but that may be attributed to the fact that these signs are a bit different from the German signs. For example the 30km sign also includes text underneath.

![alt text][image6]
