# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_images.png
[image10]: ./output_images/rgb_colour_space.png
[image11]: ./output_images/hls_colour_space.png
[image12]: ./output_images/hsv_colour_space.png
[image13]: ./output_images/ycrcb_colour_space.png
[image21]: ./output_images/hog_car.png
[image22]: ./output_images/hog_notcar.png
[image25]: ./output_images/scaled_features.png
[image30]: ./output_images/all_windows.png
[image31]: ./output_images/car_windows.png
[image32]: ./output_images/car_windows1.png
[image40]: ./output_images/heatmap.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for feature extraction is contained in code cells 1 to 7 of the IPython notebook [project.ipynb](./project.ipynb).

I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I explored different colour spaces to see which might offer the best separation for pixels from cars vs. non-cars:

| ![alt text][image10] | ![alt text][image11] |
| ![alt text][image12] | ![alt text][image13] |

I also explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:

![alt text][image21]
![alt text][image22]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and then looked at the classification accuracy using each feature set. Here are the results:

| Colour space | HOG channel(s) | Orientations | Pixels per cell | Cells per block | Accuracy |
| --- | --- | --- | --- | --- | --- |
| YCrCb | All | 9 | 8 | 1 | 0.9901 |
| HSV | All | 9 | 8 | 1 | 0.9868 |
| YCrCb | All | 9 | 8 | 2 | 0.9859 |
| YCrCb | All | 8 | 8 | 1 | 0.9856 |
| YUV | All | 9 | 8 | 1 | 0.9856 |
| LUV | All | 9 | 8 | 1 | 0.9831 |
| HLS | All | 9 | 8 | 1 | 0.9828 |
| RGB | All | 9 | 8 | 1 | 0.9735 |
| YCrCb | 0 | 9 | 8 | 1 | 0.9688 |
| YCrCb | 1 | 9 | 8 | 1 | 0.9412 |
| YCrCb | 2 | 9 | 8 | 1 | 0.9417 |

From this I chose the top set of parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.svm.LinearSVC()` in code block 8 of the IPython notebook. 

First I extracted HOG, histogram and spatial binned features. Then I used `sklearn.preprocessing.StandardScaler` to normalise all of the features to prevent one feature set from overpowering another. An example is given below:

![alt text][image25]

Once I had a normalised feature set I used `sklearn.cross_validation.train_test_split()` to create  a training and validation data set and then used the `sklearn.svm.LinearSVC.fit()` function to train the classifier.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use an overlap of 75% for my search windows and used the following pixel sizes:

* 48x48
* 64x64
* 96x96
* 128x128

I used smaller search windows near the road horizon and then used larger windows closer to the camera. The set of windows can be seen below with the 128x128 shown in white and the 48x48 shown in red:

![alt text][image30]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are a couple of examples of the window detection in action: 

![alt text][image31]
![alt text][image32]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_videos/project_output.mp4)

I added the lane detection pipeline from the previous project so that I had `VehicleTracker` and `LaneTracker` classes. I then created a composite `CombinedTracker` class which instantiated both other trackers and carried out both sets of processing on the project image.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap, thresholding and resulting bounding boxes:

![alt text][image40]

To help reject false positives the bounding boxes for each frame were collected and then I calculated a heatmap using bounding boxes from multiple frames. This helped to significantly reduce the number of false positives in the video.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that even with a significant amount of averaging I was still seeing occasional false positives. In addition the bounding box size is not particularly tight to the cars.

I wonder if a CNN deep learning approach to the classification would provide better results. Additionally combining the camera data with radar/lidar would probably improve the performance significantly.

I also wouldn't necessarily expect my current approach to be particularly robust to changes in lighting/weather.

I made an attempt at tracking the bounding boxes to smooth them out. However I think given more time a Kalman filter approach could be used to smooth out the detections and reject outliers.