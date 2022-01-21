University of Alberta
Department of Computing Science
MSc with Specialization in Multimedia
Winter Semester 2022
MM805 Computer Vision
Programming Assignment
Assigned on January 12
Deadline by Feb 12

Implementation can be done in Python (OpenCV) or Matlab, along with a report to describe what you did and include the results. 

Q1. (40 points) Feature extraction and matching: (use the images from https://sourceforge.net/projects/adobedatasets.adobe/files/adobe_panoramas.tgz/download)
    a. (10 points) Select and implement one of the point feature detectors we have explained or use other methods you learned about. (Make sure to implement the feature detector yourself). Explain your selected detector and show the results. 
    b. (20 points) Implement a simple feature matching by using two feature descriptors of your choice (you can use the available feature descriptors in OpenCV or Matlab). Compare the two feature descriptors and the matching results on a few different images.
    c. (10 points) Instead of finding feature points independently in multiple images and then matching them, find features in the first image of a video or image sequence and then re-locate the corresponding points in the next frames using search and gradient descent (Shi and Tomasi 1994). When the number of tracked points drops below a threshold or new regions in the image become visible, find additional points to track.

Q2. (40 points) Optical flow: (use the motion sequences available at https://vision.middlebury.edu/flow or http://sintel.is.tue.mpg.de)
Compute optical flow between two images.
    a. (15 points) Implement the Lucas-Kanade algorithm.
    b. (25 points) Visualize the optical flow in two video sequences, provide some examples where Lucas-Kanade fails. Explain the reason in each case.

Q3. (20 points) Head detection: 
You are asked to implement an algorithm that can detect head in images and videos. The algorithm should detect head regardless of viewing direction of the camera. What is your suggestion? Try to eliminate false positives as much as you can.