# ResJND
The official code and dataset of our paper:

Rethinking and Conceptualizing Just Noticeable Difference Estimation by Residual Learning.

# CPL-Set

CPL-Set consists of 882 pristine images and their corresponding CPL images, selected through carefully designed subjective experiment.

 ![Image](https://github.com/Knife646/ResJND/blob/main/figure/CPL-set.png)
 

# ResJND Pipeline
Pipeline of our proposed ResJND.

Overall, our ResJND model contains two data flows: the identity flow and the residual flow. We adopt Global Residual Learning (GRL) in the identity flow by adding the input image to the output of the last convolutional layer and introduce recursive learning into the residual flow by constructing the recursive block structure.

It takes the original image as its input and predicts the corresponding CPL image as the output. The parameterization of the convolution layer is indicated as ”input channel → output channel.” The ultimate JND map is derived by computing the RMS of the intermediary residual map denoted as ***R***.

 ![Image](https://github.com/Knife646/ResJND/blob/main/figure/ResJND.png)

# Resource
If you need the CPL-Set and ResJND for academic usage, you can download through follow link:
