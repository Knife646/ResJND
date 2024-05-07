from data_loader import test_data
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from model.CPL_Net import CPL_Net

"""
You can use this code to generate JND map.
"""

def rgb_to_rms_gray(img_array):

    r, g, b = img_array[0, :,: ], img_array[1, :, :], img_array[2, :, :]
    gary_img = np.sqrt((r ** 2 + g ** 2 + b ** 2) / 3)

    return gary_img


parser = argparse.ArgumentParser(description="PyTorch ResJND Test")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--cuda", default="true", help="Use cuda?")
opt = parser.parse_args()

# Load model
model1 = CPL_Net()
model1.load_state_dict(torch.load('checkpoint/model_epoch_250_44.9119873046875.pth'))
model1.cuda()

# Dataload
test_set = test_data(test_path=f"example"
                     ,label_path="example")
test_data_loader = DataLoader(dataset=test_set, batch_size=opt.batchSize,shuffle=False)

k=1
metric_rmse = []
for data in test_data_loader:
    img,input_img,label,name = data
    input_img = input_img.cuda()
    CPL = model1(input_img)
    Cpl = CPL.detach()

    name = str(name)
    name = name.split(".")[-2]
    name = name.split("\\")[-1]

    # Get CPL in numpy
    Cpl = Cpl.cpu()
    Cpl = Cpl.squeeze(0)
    Cpl = np.array(Cpl)
    Cpl = np.float32(Cpl)#CHW

    # Get reference in numpy
    img = img.cpu()
    img = img.squeeze(0)
    img=np.array(img)
    img=np.float32(img)

    # Get residual map as JND map
    JND = np.abs(Cpl - img)
    JND = JND*255
    JND = rgb_to_rms_gray(JND)

    # Normalization
    JND_Norm = (JND)/(np.max(JND))

    # Save JND map
    cv2.imwrite(f"result/{name}.bmp", JND_Norm*255)





