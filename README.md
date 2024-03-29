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
[image7]: ./output_images/accuracy.png "Accuracy plot"

## Implementation

This project was implemented within the `Traffic_Sign_Classifier.ipynb` jupyter notebook . The dataset images, where downloaded from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads). Currently, it was possible to download a dataset for trainning, validation and testing. The pickle files can be found under the traffic-sign-data folder. The augmented images can be found under the data folder.

## 1. Loading the Data

The code for this step is contained in section 1 of the IPython notebook `Traffic_Sign_Classifier.ipynb`. For each pickle file, I created two arrays one for the features and one for the coresponding labels. Section 1.1 of the IPython notebook includes some statistics for the provided data. The class names are read from the file `signnames.csv`. In section 1.2 of the jupyter notebook, a random traffic sign for each class is displayed, along with its corresponding class label attached.

![alt text][image1]

A histogram of the distribution of the various samples per class is presented below.

![alt text][image2]

## 2 Model Architecture Design and Test

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson was used as a starting point with minor modifications. The number of classes was changed from 10 to 43 and the grayscale input to color. Additionally, `scope` labels where added in order to utilize Tensor Board for better visualization of the network performace. Tensor board allows for better visualization of the effects that various hyper parameter changes have on the trainned model. The graph of the implemented network can be seen in the figure below.

![alt text][image3]

If the jupyter notebook is executed, an interactive graph can be obtained by typing `tensorboard --logdir=/tmp/udacity/graphs/traffic-sign-classifier-{timestamp}, pasting the resulting url on a browser. 

The initial validation accuracy, of the LeNet-5 model was approximatelly 0.89, using the original dataset. Augmenting and normalizing the data increased the validation accuracy to 0.93.
The optimizer was also changed from SGD to Adam and various combinations of hyper parameters were tested:

* learning rates 0.0010, 0.0015, 0.0020
* batch sizes 32, 64, 128 ,256
* epochs 10, 15,20, 25, 30
* dropout keep probability 0.5, 0,6, 0,75, 0.8

In addition to that, learning rate decay was also tested but later removed (commented out) as it gave worse results than expected. L2 regularization was also tested, using a penalty factor of 5e-4, but that also did not improve performance and was subsequently removed. As the difference between trainning and validation accuracy was quite big, dropout layers were added to the original lenet model, to reduce overfitting. Initially, a dropout layer was added after each convolutional and fully connected layer, but this reduced all accuracies. Imprevement was noticed when dropout layers were added only after the fully connected layers. Various keep probabilities were tested, in conjunction with different batch sizes, epochs and a smaller learning rate. It was found that smaller batch sizes performed better, especially after the addition of dropout layers to the model.
The best results were obtained for a learning rate of 0.0010, batch size 64, keep probability of 0.75 trainning for 15 epochs. The test accuracy achieved was 0.961. A plot of the trainning and validation accuracy (per epoch) can be seen in the figure below:

![alt text][image7]

### 2.1 Data pre-processing

The first pre-processing technique used, was normalization. However, there was only a marginal improvement in accuracy. I then converted the image data sets to grayscale. The implementation can be seen within method `preprocess_dataset()` found in section 2.1 of the jupyter notebook. This improved the accuracy (0.90), however performing an adaptive histogram equalization on the entire dataset increased the accuracy even more and gave more concistent results. This is implemented within method `normalize_images()`. Adaptive histogram equalization effectively reduces the effect that bad light conditions (or capture settings of the camera) have on the quality of the image, revealing more details. I also tried to merge these two methods, but the result was actually worse. Eventually, only the later method was used. The implementation for both methods, can be found in section 2.1 of the jupyter notebook. The normalized images were saved in a pickle file. The code checks for the existence of this file in order to avoid re-computing the histogram equalization on subsequent executions of the notebook, unless the corresponding file is removed.
In addition to that, the data set was augmented by additing rotated and shifted images. Initially, the implementation was based on scikit, but it was slow and lengthy. I replaced this code with the `ImageDataGenerator()` method available through the Keras library. I was also able to add zoomed images. The trainning dataset was saved into a pickle file `augmented_data.p` in the output_images folder. A total of 258000 images were used for trainning and 32250 for validation.

### 3 Testing the model on new images

Ten images from the Belgian traffic sign set were downloaded, as they are pretty similar to the German traffic signs. These are reference images i.e. taken from a document. The class names were read from signnames.csv as it contains mappings from the class id (integer) to the actual sign name. The images can be seen on the figure below:

![alt text][image4]

Since these were reference images the classification was perfect as can be seen from the figure below:

![alt text][image5]

A second set of images, taken from actual pictures was also used to test the classification accuracy when real world image are used. These traffic signs were from a French traffic sign set. Some of the signs were missclassified but that may be attributed to the fact that these signs are a bit different from the German signs. For example the 30km sign used also includes text underneath.

![alt text][image6]

Plotting the probabilities per graph, we observe that for the the correctly predicted signs we only get one probability (of significant size). For incorrectly classified images we observe that there are more than one. Classification failure may be attributed to several reasons.

* This can be a difference in signs used, as in the case of the 30km zone sign.
* In other cases when excessive shrinking is applied to the test image, as in the cases of the roadworks sign used, it becomes similar to more than one signs, such as childen crossing.
* Sometimes, shrinking images that are normally not square, into a 32x32 square image, changes their aspect ratio, distorting them significantly. For example the 30km sign used for trainning is square. But the 30km sign used for testing is not, as it contains the zone keyword.
* Signs when photographed from an angle, such as the 30km, 60km and 80km look identical, especially, if the test image is shrunk significantly.
* Tilt angle, also effects classification significantly, as it makes certain details look less prominent. For example the roadworks sign, when titled to the right results in the human being more pronounced while the details to the right dissappear, making the sign look like a dangerous curve to the left sign. Number sign are particularly effected when photographed with large tilt angles
* Contrast is normally a huge issue, but in our case we chose a pre-processing technique (adaptive histogram equalization) that reduces this effect
* Background objects can also play a significant role, especially if they contain other traffic signs.
* Jiter also makes certain details look less prominent, particularly for signs that have a lot of detail i.e. childern crossing classified as road norrows to the left/right sign.

Finally, we calculate the real world accuracy by counting correct predictions vs total number of images. In this case the real world accuracy is calculated to be 0.875% i.e. one out of 8 images is incorrectly classified. The html file for this notebook can be found in the root folder.

### 4 Discussion

The classifier worked reasonably well, however, most of the hyper parameter tuning was done manually by trial and error. At a later stage Tensor Board visualizations were incorporated. If the code was modified to test the various different parameters on the same run, such as different learning rates, or learning rate decay, Tensor Board graphs could be used to highlight the differences and quite possibly choose a better combination of hyper parameters. In addition to that, with the introduction of embeddings it is possible to implement advanced visualizations. Due to the lack of online examples this required an enormous effort and it was therefore not implemented.

