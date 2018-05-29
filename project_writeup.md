## Vehicle Detection Project Writeup
### This report documents the solution process for detecting vehicles from a dashboard mounted camera video stream.

---

**Vehicle Detection Project**

The goals / steps of this project include:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier.
* As desired, apply color transforms and append binned color features, as well as histograms of color, to the HOG feature vector.
* Normalize features and randomize a selection for training and testing from the input data set.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected on the video stream.

[//]: # (Image References)
[image1]: ./output_images/data_visualization.png
[image2]: ./output_images/hog_image_vis/hog_visualization003.png
[image3]: ./output_images/hog_image_vis/example_car_image_002.png
[image4]: ./output_images/sliding_windows_vis001.png
[image5]: ./output_images/test_image_bbox.png
[image6]: ./output_images/find_cars_heat_map_optimization_thres-2.png
[image7]: ./output_images/find_cars_heat_map_optimization_thres-0.5.png
[image8]: ./output_images/find_cars_heat_map_optimization_thres-10.png
[image9]: ./output_images/find_cars_heat_map_images.png
[video1]: ./output_videos/test_output.mp4
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### The rubric points are individually considered and solution approaches described for how each point was addressed in the project implementation.  

---
### Project Writeup

### Histogram of Oriented Gradients (HOG)

#### 1. The HOG function is in the Analysis Pipeline function section (P5_Vehicle_Detection.ipynb, Cell 5).

First, the images for 'vehicles' and 'non-vehicles' were read into lists cars and notcars respectively. The number of images in each list were printed out, along with sample images for Car and Not Car:
* vehicle images:  8792
* non-vehicle images:  8968
* image shape = (64, 64, 3)

![Car and Not Car samples][image1]

Functions were defined for color histogram, spatial binning, then the extract_features() and single_img_features() functions were defines, which referenced bin_spatial() and color_hist() functions.

Key parameters for HOG included color space, orientations, pixels_per_cell, cells_per_block and hog_channels. A variety of parameters combinations were evaluated in informal testing, with image outputs for car and notcar images saved to compare to one another.

![HOG image hog_visualization003][image2]

![HOG image example_car_image_002][image3]

Of particular interest is the way that car and notcar images are represented. Generally a car has a characteristic shape with the outer body and license plate providing a characteristic pattern. Environment images (notcar) typically are represented by horizontal or uniformly oriented stripe patterns. This difference in features should then provide a good basis for being able to classify the car images from the environment. The following HOG parameters provided good visual differentiation between car and notcar images:
* orient = 8-10
* pix_per_cell = 6-8
* cell_per_block = 2

#### 2. Final choice of HOG parameters

Final tuning of HOG parameters was done during the training of the SVC Classifier. The orient, pix_per_cell, and cell_per_block values were tuned previously based on the visual results. The color_space and hog_channel were evaluated with the error of the classifier training.

##### Defined features parameters:
* color_space = 'YCrCb'
* orient = 10
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL'
* spatial_size = (32, 32)
* hist_bins = 64
* spatial_feat = True
* hist_feat = True
* hog_feat = True

A test accuracy of 98.93 % was achieved with the listed HOG parameters, which served as the indication that they represented a good set of values.

#### 3. Linear SVM Classifier

The linear SVM was trained using the previously described HOG parameters. Initially a random sample of 1000 data points was used when evaluating HOG parameters. After the ideal combination was achieved with high test accuracy the full data set was trained. A test size of 0.1 was specified in the train_test_split function of sklearn.

### Sliding Window Search

#### 1. Sliding window search (Cell 13-19) was implemented to identify vehicles.

The sliding window search included a search region definition in order to reduce computation time per image. This localization region was defined as the region between the hood of the car to the horizon of the road, where the back of a vehicle would logically exist (from the perspective of the dashboard camera).

![alt text][image4]

In the sliding windows function, the window search would move across the x dimension and y dimension of the image (with origin at the upper left corner). These parameters were defined as:

* x_start_stop = [600, None]
* y_start_stop = [380, 680]

The size of the window and the degree of overlap as the window moves across the image were defined as follows:

* xy_window=(128,128)
* xy_overlap=(.7, .7)

The overlap parameter was evaluated between 0.5 to 0.8, with optimal vehicle identification occurring with an overlap of 70% (0.7).

The sliding window function was evaluated on test images, with characteristic results shown below. In the search_windows() function, a list of found windows is defined, if a window identifies as containing a vehicle (which occurs if the prediction of the trained model is equal to 1), then the on_windows list is appended, and the draw_boxes() function called to draw a green box around the identified image region (which should include a vehicle). To improve the vehicle finding, the double boxes will needed to be accounted for.

![alt text][image5]

#### 2. Pipeline Flow

From windows search to the final bounding box overlay, a heatmap was implemented in order to reduce false positives and the multiple bounding boxes over a single car as shown previously.

![alt text][image6]

The heat map addresses the problem of recurring detections, and frame by frame it is used to reject outliers and follow detected vehicles. The heatmap function includes a threshold so that areas outside of the detected region is black, while the center of the detection area is lighter. The threshold value was tuned in order to localize the vehicle region. Examples of low (0.5) and high (10) thresholds are shown below:

![alt text][image7]
![alt text][image8]

### Video Implementation

#### 1. Test and Project video outputs

The video outputs are included here, with bounding boxes displayed with detected vehicles.

[link to Test Video result](https://youtu.be/NSmyolgroNE "Test Video")


[link to Project Video result](https://youtu.be/5x_WFBsFStA "Project Video")


---

### Discussion

#### 1. Current Vehicle Detection Approach

In a relatively well-controlled environment, the use of feature characterization based on color transforms, HOG, etc. seems to work relatively well. however, the training data for this project was taken from the most ideal conditions possible: a sunny day on a highway in California. Visually this provided the best case, given the high color saturation of the images pulled from the video.

This project represents the easiest environment for vehicle detection, but in reality the algorithm would need to perform well in a variety of conditions, and problems will likely arise with lower contrast environments, such as overcast days, early morning, and at night, where it will be much more difficult to cleanly derive image features using HOG or color transformation methods. A better approach to data ingestion could be investigated here, applying transformations in OpenCV for better normalization, gray scale, and sharpening to see how this could improve the feature extraction and HOG visualization to differentiate vehicles from the environment.

#### 2. Lane Finding Integration

Currently there is only information about vehicle detection presented, but there is no context for the way the program perceives the local environment, as was accomplished in the CarND Advanced Lane Finding project, or the Behavior Transfer project based on CNN characterization of driving behavior. This could give context to vehicle detection, by defining the search region based on lane geometry.

#### 3. Neural Network Approach

Previous work with the use of Convolutional Neural Networks (CNN) has shown good performance in training for object recognition based on visual training data. This was used for Traffic Sign Classification, and for training driving behavior. This could aid in vehicle classification as well as characterizing the environment, which would potentially encapsulate the benefits of combining Lane Finding with Vehicle Detection as described above.
