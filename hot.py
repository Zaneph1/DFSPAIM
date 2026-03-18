# coding: utf-8

import os
import cv2
import numpy as np

from models import DenseNet121_Melanoma


import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
resume_path = os.path.join(BASE_DIR, r"weight\dpn\9c_meta_b3_768_512_ext_18ep_best_20_fold0.pth")
# Input image path
single_img_path = os.path.join(BASE_DIR, r"model_pic\\")
# Output heatmap path
save_path = os.path.join(BASE_DIR, r"0 (1).jpg")


# image = cv2.imread(single_img_path+"0 (1).jpg")
# # print(single_img_path+"0 (1).jpg")
#     # img = Image.open(img_path).convert('RGB')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # image = image.astype(np.float32)
# #
# # image = image.transpose(2, 0, 1)
# # img = torch.tensor(image).float()
# img = preprocess_image(image, mean=[0.485, 0.456, 0.406],
#                                              std=[0.229, 0.224, 0.225])
#
# print(img.shape)
#     # if transform:
#     #     img = transform(img)
#     # img = img.unsqueeze(0)  # (1, 3, 448, 448)
#     # img = get_trans(img, 5)

model = DenseNet121_Melanoma(
            'densenet121',
            out_dim=6
        )
state_dict = torch.load(resume_path,map_location=torch.device('cpu'))
state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
model.load_state_dict(state_dict, strict=True)
model.eval()

# model = resnet50(pretrained=True)
from pytorch_grad_cam.utils.model_targets import  BinaryClassifierOutputTarget
# target_layers = [model.enet.blocks]

target_layers = [model.enet.features]
# input_tensor = img# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers,)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.
# for i in range(0,5):
#     for j in range(1,6):
#         print(type(str(single_img_path+str(i)+" ("+str(j)+").jpg")))
#         a = str(single_img_path+str(i)+" ("+str(j)+").jpg")
#         image = cv2.imread(a)
#         print(single_img_path+str(i)+" ("+str(j)+").jpg")
#         # img = Image.open(img_path).convert('RGB')
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # image = image.astype(np.float32)
#         #
#         # image = image.transpose(2, 0, 1)
#         # img = torch.tensor(image).float()
#         img = preprocess_image(image, mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#
#         print(img.shape)
#         input_tensor = img
#
#         targets = [ClassifierOutputTarget(i)]
#         print("这是第",i,j)
#
#         # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
#         grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
#
#         # In this example grayscale_cam has only one image in the batch:
#         grayscale_cam = grayscale_cam[0, :]
#         image = cv2.imread(a)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = image.astype(np.float32)/ 255.0
#         visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
#
#         # You can also get the model outputs without having to re-inference
#         model_outputs = cam.outputs
#         image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
#         cv2.imwrite("xception_"+str(i)+str(j)+".png", image)
#         print(model_outputs)

single_img_path = os.path.join(BASE_DIR, r"model_pic\\0\\")

a = str(single_img_path+"_18-03822_RGB_Extended0_256_1black.jpg")
# a = str(single_img_path+"_19-09867-43-44_RGB_Extended0_256_1black.jpg")
# a = str(single_img_path+"_18-11013-2_RGB_Extended1000_256_22black.jpg")
# a = str(single_img_path+"_18-19659_RGB_Extended11_256_1black.jpg")
# a = str(single_img_path+"_18-00888_RGB_Extended7_256_1black.jpg")
# a = str(single_img_path+"_16-14762_RGB_Extended5_256_1black.jpg")

image = cv2.imread(a)
# print(single_img_path+str(i)+" ("+str(j)+").jpg")
# img = Image.open(img_path).convert('RGB')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = image.astype(np.float32)
#
# image = image.transpose(2, 0, 1)
# img = torch.tensor(image).float()
img = preprocess_image(image, mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

print(img.shape)
input_tensor = img

targets = [ClassifierOutputTarget(0)]
# print("这是第",i,j)

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
image = cv2.imread(a)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32)/ 255.0
visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

        # You can also get the model outputs without having to re-inference
model_outputs = cam.outputs
image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
cv2.imwrite("dpn_"+"0.png", image)
print(model_outputs)