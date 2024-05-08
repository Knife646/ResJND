# ResJND
The official code and dataset of our paper:

Rethinking and Conceptualizing Just Noticeable Difference Estimation by Residual Learning.

# CPL-Set

CPL-Set consists of 882 pristine images and their corresponding CPL images, selected through carefully designed subjective experiment.

 ![Image](https://github.com/Knife646/ResJND/blob/main/figure/CPL-Set.png)
 
If you need the CPL-Set and ResJND for academic usage, you can download through follow link:

[Download](https://pan.baidu.com/s/1QbDY4u-q1CIqqKHr1wu0fg?pwd=egqh)

# ResJND Pipeline
Pipeline of our proposed ResJND.

Overall, our ResJND model contains two data flows: the identity flow and the residual flow. We adopt Global Residual Learning (GRL) in the identity flow by adding the input image to the output of the last convolutional layer and introduce recursive learning into the residual flow by constructing the recursive block structure.

It takes the original image as its input and predicts the corresponding CPL image as the output. The parameterization of the convolution layer is indicated as ”input channel → output channel.” The ultimate JND map is derived by computing the RMS of the intermediary residual map denoted as ***R***.

 ![Image](https://github.com/Knife646/ResJND/blob/main/figure/ResJND.png)

# Training and Generating
Uncompress the dataset and put it in the working directory:
 
    ResJND
       ├──train.py
       ├──test.py
       ├──data_loader.py
       └──Dataset
             ├──train
             ├──test
             ├──valid
             └──get_patch

If you want to train ResJND, just execute the following command:

     train.py

If you want to train on your own dataset, you can use ''gete_patch.m'' tool to crop the image to make the input dimensions same.


The checkpoint will be saved to ''/checkpoint''

If you want to generate JND maps, just execute the following command:

     test.py

The generated JND maps will be saved to ''/result''

Note that the key environment used in our experiments are:

     python == 3.7
   
     torch == 1.13.1 + cu117
   
     torchvision == 0.14.1
   

# Citation
If you find this repo useful, please cite our paper.

    ''Rethinking and Conceptualizing Just Noticeable Difference Estimation by Residual Learning.''
