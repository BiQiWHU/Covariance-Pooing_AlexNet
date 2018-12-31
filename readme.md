In this project, we implement the algorithm MSCP in the paper *Remote Sensing Scene Classification Using Multilayer Stacked Covariance Pooling* on UC Merced dataset.
In this framework, we get the tensors' of the 5th convolutional layer in the AlexNet and use  the MSCP method to implement feature reduction, then send these features into a SVM classifier.
In the paper, it is reported that via using this method, the classification accuracy on UCM dataset can achieve 0.96 while in this project, with a similar hyperparameter setting, we can achieve an accuracy about 0.84 to 0.86.

Requirement:
Python3.X
OpenCV3
Tensorflow>1.5
Numpy

The implementation procedures are listed as follows:
(1) Download this project
(2) Copy the UC Merced dataset into the file, and do pay attention to the file folder.
(3) Run tfdata.py to generate your train and test data, with a tfrecord format.
(4) Run training.py to train the AlexNet and get the trained model.
(5) Run trainSVM.py, it will implement the MSCP and print the test accuracy.

Enjoy!

