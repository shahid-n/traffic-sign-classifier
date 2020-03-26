# **Convolutional Neural Network for Traffic Sign Recognition** 

**Traffic Sign Recognition Project**

The goals of this project are the following.
* Load the data set (see below for links to the project data set)
* Explore, summarise and visualise the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyse the softmax probabilities of the new images
* Summarise the results with a written report


[//]: # (Image References)

[samples]: ./output_images/sample_signs.png "Visualisation"
[train]: ./output_images/hist_train.png "Training Images"
[valid]: ./output_images/hist_validation.png "Validation Images"
[test]: ./output_images/hist_test.png "Testing Images"
[newimgs]: ./output_images/new_images.png "Some New Test Candidates"
[sftmax1]: ./output_images/softmax_1.png "Prediction #1"
[sftmax2]: ./output_images/softmax_2.png "Prediction #2"
[sftmax3]: ./output_images/softmax_3.png "Prediction #3"
[sftmax4]: ./output_images/softmax_4.png "Prediction #4"
[sftmax5]: ./output_images/softmax_5.png "Prediction #5"
[layerExplore]: ./output_images/test_sign.png "A Test Sign Used to Visualise the Feature Maps"
[layer1]: ./output_images/plot_1.png "Layer 1 Feature Maps"
[layer2]: ./output_images/plot_2.png "Layer 2 Feature Maps"
[layer4]: ./output_images/plot_3.png "Layer 4 Scores"
[layer5]: ./output_images/plot_4.png "Layer 5 Scores"
[logits]: ./output_images/plot_5.png "Logits"

## Rubric Points
### The [rubric points](https://review.udacity.com/#!/rubrics/481/view) are considered individually within this section to elaborate on how the implementation covers each one.  

---
### Writeup / README

This readme addresses the write-up requirement of the rubric. The [project code](https://github.com/shahid-n/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb) is the key means of addressing all the other portions of the rubric.

### Data Set Summary & Exploration

#### 1. High level summary

The first step was to load the German traffic sign dataset and to use Numpy to calculate some statistics which are summarised below.

* The size of the _original_ training set was 34799
* The size of the _augmented_ training set was 111356
* The size of the validation set was 6960
* The size of test set was 12630
* The shape of a traffic sign image is 32 x 32 pixels, with 3 colour channels (RGB)
* The number of unique classes/labels in the data set is 43

#### 2. Example signs and visualisation of the distribution of the images

The figure below shows a random sampling of the signs along with their associated labels, which are part of the data set.

![alt text][samples]

The next three figures show the frequency distribution of the 43 different kinds of images found in the training, validation and testing subsets, respectively.

![alt text][train]
![alt text][valid]
![alt text][test]

### Design and Testing of the Model Architecture

#### 1. Preprocessing, Normalisation and Augmentation

The first step was to convert the images to greyscale in order to improve the robustness of the network with respect to changes in colouration or shading, whilst simultaneously -- and implicitly -- placing increased emphasis on shapes and lines as the keys to proper sign recognition.

Next, the images were all normalised, as this improves both the efficacy and ultimate accuracy of the trained network.

Last but not least, the training data were augmented with 3 different rotated versions of the originals, thus quadrupling the size of the training set.

#### 2. Model Architecture Summary

The final model comprises the following layers.

|    Layer              |     Description                            | 
|:---------------------:|:------------------------------------------:| 
| Input         	    | 32x32x3 RGB image                          |
| Convolution, 5x5      | 1x1 stride, valid padding, output 28x28x6  |
| ReLU                  | Rectified linear activation                |
| Max Pooling, 2x2      | 2x2 stride, valid padding, output 14x14x6  |
| Convolution, 5x5      | 1x1 stride, valid padding, output 10x10x16 |
| ReLU                  | Rectified linear activation                |
| Max Pooling, 2x2      | 2x2 stride, valid padding, output 5x5x16   |
| Convolution, 2x2      | 2x2 stride, valid padding, output 2x2x100  |
| ReLU                  | Rectified linear activation                |
| Flatten to 1-D        | Output 400                                 |
| Fully connected       | Output 120                                 |
| Dropout               | Keep probability: 0.6                      |
| Fully connected       | Output 84                                  |
| Dropout               | Keep probability: 0.54                     |
| Fully connected       | Output 43                                  |


#### 3. CNN Training Procedure

The fundamental training approach chosen for this convolutional network was to minimise the mean cross entropy by using the Adam optimiser.

Whilst the default hyperparameter values provided with the LeNet code example proved to be surprisingly good initial guesses, some tweaks were necessary to achieve at least 93 % validation accuracy. In particular the number of epochs had to be tuned in conjunction with the keep probability for dropouts that were introduced into a couple of layers in the modified LeNet architecture.

The images were normalised to have zero mean and converted to greyscale in order to ensure greater robustness of the training regimen to variations in colour and shading, whilst simultaneously emphasising the importance of shapes and lines to ensure successful classification of traffic signs.

Last but not least, the data set was augmented by rotating the images, initially in integer multiples of 90 degrees. However, it quickly became apparent that this quick and dirty way of quadrupling the training set had a few drawbacks, such as the model having difficulty distinguishing between images that match an existing valid image from the data set after a +/- 90 degree rotation -- the Keep Left and Keep Right signs are a good example of this problem. Consequently, the final training set used rotations of +/- 25 and 180 degrees, respectively, all of which were augmented to the original group of training images.

#### 4. Architecture Tweaks and Parameter Tuning

An iterative approach was chosen to try different combinations of hyperparameter values, ultimately followed by some architecture modifications and further fine-tuning of the parameters.

Initially the standard LeNet architecture was chosen because of its proven capabilities in image classification applications. Nevertheless, although the model was able to achieve very high accuracy during training, validation accuracy was somewhat poorer, which pointed to issues with overfitting and potential limitations in terms of generalising to new sets of traffic signs in realistic driving scenarios.

Consequently, the architecture was adjusted by introducing dropouts in two of the layers, and furthermore, an extra convolutional layer was added immediately prior to the fully connected layers, which led to markedly improved accuracy whilst simultaneously avoiding overfitting.

The final model performance statistics are as follows.
* training set accuracy of 0.943
* validation set accuracy of 0.977 
* test set accuracy of 0.912

The above results suggest the model might have undergone slight overfitting; nevertheless, these results represent an improvement over some early results of 0.938 training accuracy and 0.85 test accuracy.

### Test Results on New Images

#### 1. Performance on New Images from the Web

Below are five new sample images downloaded from the web. Note that they are in colour, and since the network supports 3 channels despite only being trained using greyscale images, they could be fed in directly without the need of any pre-processing, save for down-sampling them to the compatible pixel dimensions of 32 x 32.

![alt text][newimgs]

#### 2. Model Predictions

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80 %. This is obviously below the test set accuracy of 91 %, but can be accounted for by the small sample size.

Nevertheless, the network's inability to distinguish between the 60 and 30 kph signs is an unexpected surprise, since the leading digits are quite different. It is possible, however, that some of the speed limit images are too blurry or obscured in some instances; alternatively, it could just be an idiosyncrasy of this particular network architecture-cum-training weights & biases combination.

#### 3. Deeper Examination of the Predictions

The figures below illustrate the predictions generated by the model.

![alt text][sftmax1]
![alt text][sftmax2]
![alt text][sftmax3]
![alt text][sftmax4]
![alt text][sftmax5]

The first prediction is clearly wrong; the network thinks it is a 30 kph sign, even though the figure is clearly 60. All the other correct predictions returned probabilities close to 1. The accuracy result of this test is `4/5 = 80 %`.

The output copied below from the Jupyter notebook shows the top 5 predictions with associated labels corresponding to each of the 5 images above, respectively.

    TopKV2(values=array([[9.96405900e-01, 3.59406671e-03, 2.23206946e-08, 1.16978606e-11,
            7.45939730e-18],
           [9.99679804e-01, 3.20266816e-04, 7.91848087e-10, 2.36586112e-10,
            1.46885074e-10],
           [9.92601573e-01, 6.23094290e-03, 1.12409412e-03, 2.72912057e-05,
            6.72608076e-06],
           [9.30282474e-01, 6.97153732e-02, 8.38632957e-07, 6.83138637e-07,
            1.94384668e-07],
           [1.00000000e+00, 2.21564802e-20, 4.97184035e-23, 3.04714904e-26,
            2.07820506e-26]], dtype=float32), indices=array([[ 1,  3,  5,  2,  0],
           [23, 19, 13, 31, 20],
           [41, 42, 32, 12, 10],
           [12, 42, 30, 41, 38],
           [18, 26,  1, 27, 25]], dtype=int32))

### Visualising the Feature Maps

The test image below was used to illustrate the feature maps and excitations within each layer of the network. Note that this is a colour image being fed into a network that was only trained on greyscale images (albeit with support for 3 channels, which all contained the same greyscale training data).

![alt text][layerExplore]

The layer visualisations are shown below. Layer #3, whose output shape is 2x2x100, has been omitted due to space limitations precluding a suitable or useful presentation of every feature map from that particular set of outputs.

![alt text][layer1]
![alt text][layer2]
![alt text][layer4]
![alt text][layer5]
![alt text][logits]

---
## Concluding Remarks

The main conclusion of this project is the need for a judicious approach to the overall design of a deep neural network with due consideration given to all of the following aspects.

1. network architecture
2. coarse parameter tuning
3. iterative fine-tuning of parameters in conjunction with architecture modifications
4. isolation of test data from the validation and training sets
5. statistically sound image pre-processing and augmentation methods
6. reguralisation strategy such as introduction of dropouts

Due to time constraints, point #5 above did not receive the level of attention it deserves, which needs to be rectified in the future as part of ongoing improvements.

Specifically, when we examine the histogram plots from the `Data Set Summary & Exploration` section, it is quite apparent that the distribution of images is not uniform. Furthermore, the method used to augment the data was somewhat crude, as it simply involved rotating the images through 4 different fixed angles. Although somewhat more time and resource intensive, a better approach would be to run a separate offline routine with the ability to apply arbitrary -- and perhaps randomised -- amounts of rotation on a randomly sampled subset of the training data, in addition to performing slight colour modifications, translations, skew or other shape transforms, blurring, addition of random noise or other minor artifacts, etc. This augmented training set could then be fed into the network to achieve a fundamentally more robust model.

I expect the techniques suggested above would lead to improved performance, especially on new real world images. As such, the initial LeNet architecture was quite adept at achieving a high training accuracy in the 95 to 98 % range, but would then underperform to the effect of returning only around 82 to 88 % accuracy on the validation set, an average difference of over 10 %, which suggests overfitting.

The design, parameter tuning and pre-processing choices outlined in the prior sections helped to narrow the gap so that, whilst the training accuracy levelled off around 94 % due in part to the large and more varied training set, the validation and test accuracy metrics nevertheless reveal promising potential for this model to be applied in a variety of real world scenarios, where it could be reasonably expected to return an accuracy of at least 90 to 91 %.
