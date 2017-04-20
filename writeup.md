**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to  collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/Center_lane_driving.JPG "Example for the center lane driving"
[image3]: ./images/Training_MSE_epoch.JPG "Training MSE over epochs"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup report.md summarizing the results
* video.mp4 with one full lap around the track one

####2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track one by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The initial model was based on the NVIDIA CNN-architecture introduced in [NVIDIA-architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and consists of a three convolutional layers with 2x2 strides and 5x5 filter sizes  followed by two convolutional layers with 3x3 filters for feature extraction and three fully-connected layers with the depths 160, 80, 16 (model.py lines 89-142). As soon as the dimensions of the input images after cropping 3x66x320(RGB) differs from the one of NVIDIA setup (3x66x200 with the YUV color map), the width sizes of the convolutional and fully-connected layers is also different.  

The model includes RELU layers to introduce nonlinearity (code lines 101-140), and the data is normalized in the model using a Keras lambda layer (code line 94). 

####2. Attempts to reduce overfitting in the model

In order to reduce overfitting the model contains dropout layers with the keep probability 0.5 by fully-connected layers (model.py lines 131, 135 and 139). To use the dropout layers after convolution layers also more training data on other tracks are required (flipping helps not) - the vehicle drives out of the track by curves.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 80-86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The Adam-Optimizer has been applied, as it uses the momentum (moving average) in comparision to the GradientDescentOptimizer to improve the gradient direction, so the larger batch sizes can be used. Batch size has been empirically choosen equal 128.

After some tuning the hyperparameter have been set as follows: 
- learning rate: 0.001
- number of epochs: 18
- batch size: 128
- rows pixels from the top of the image: 74
- rows pixels from the bottom of the image: 20
- columns of pixels from the left of the image: 0
- columns of pixels from the right of the image: 0

(model.py lines 55-65).

####4. Appropriate training data

Following strategies for collecting of the training data has been chosen to keep the vehicle driving on the road:
1. two laps on the first track where the car drives along the center of the road as much as possible by max velocity
2. one lap on the first track where the car drives s-curves along the center of the road by max velocity
3. one lap on the second track where the car drives along the center of the road
4. one lap on the first track counter-clockwise where the car drives along the center of the road by minimal velocity
5. one lap on the first track where the car drives along the center of the road by minimal velocity to balance the training data from the counter-clockwise drive, because the vehicle has been driven more to the righton the data from 1.- 4.

###Model Architecture and Training Strategy

####1. Solution Design Approach

At the first step the convolution neural network model similar to the NVIDIA CNN-architecture introduced in [NVIDIA-architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) has been used. This model might be appropriate because it is already proven concept and shows relative good performance: 98% "autonomy" equals 5 driver interventions at the a typical drive (25 min, 12 miles) Holmdel to Atlantic Highlandsin NJ. Where the driver intervention occur when the simulated vehicle departs from the center line by more than 1 meter. 

For the current project the restrictions for the driving area are not so strength: 
- the vehicle may drive within the whole road (2 lanes) which equls approx. 7 meter, so the deviation of 2 meters from the center line are allowed. 
- two tracks given at the simulation 

Therefore it seems to be possible to train this CNN for current application (2 given tracks in simulation) with smaller amount of the training data as it has been used by NVIDIA-team.

In order to recognize the overfitting/underfitting at each epoch the input data have been divided into a training and validation set with the ratio 80:20. 

To reduce the overfitting the following steps has been performed:
- using of the dropouts by fully-connected layers
- collect more training data

The following steps can be done to improve the model:
x using of the augmented data (flipping)
x using the frames from left and right cameras for more data
x use the HLS-color space with thresholding to filter the background information

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle was very close to the border of the track: on the bridge with the dark shadow from the left and right. To improve the driving behavior in this case, more data on the brigde have been collected (slowly driven at the last recorded lap).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 89-142) consisted of a convolution neural network with the following layers of a three convolutional layers with 2x2 strides and 5x5 filter sizes  followed by two convolutional layers with 3x3 filters for feature extraction and three fully-connected layers with the depths 160, 80, 16 (model.py lines 89-142).

Here is a visualization of the architecture:
![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, first two laps on track one using center lane driving have been recorded. Here is an example image of center lane driving:

![alt text][image2]

Then the vehicle recovering from the left side and right sides of the road back to center on the first track has been recorded so that the vehicle would learn to drive to center from the lane borders. 

Then this process has been repeated on track two in order to get more data points.

After training with this data set (train:4567/ validation:1142) the vehicle was able to drive the track one without crossing the lane borders but with high oscillations within the road, also on the stright parts of the road. On the track two the vehicle can not handle curves with the bigger curvatures (approx. 1/50 m radius).

So additional data has been recorded:
- one lap on the first track counter-clockwise where the car drives along the center of the road by lower velocity
and
- one lap on the first track where the car drives along the center of the road by minimal velocity to balance the training data from the counter-clockwise drive, because the vehicle has been driven more to the right on the previous data.

After the collection process, there were __(train: 8119/ validation: 2030)__ [number of data points]. At the preprocessing step the data have been normalized in order to avoid the processing of the big values and to avoid additional bias.

Finally the data set has been randomly shuffled and __20%__ of the data has been used as a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was __18__ as evidenced by following figure:

![alt text][image3]

At the epoch 18 following values have been arrived: __Train loss: 0.0645; Validation loss: 0.0672__
That means that the model will make good predictions on both the training and validation sets.

Unfortunatelly on the track two the vehicle still can not handle spiral curves with the bigger curvatures (> as 1/50 m radius). 