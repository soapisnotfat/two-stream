"""
A sample function for classification using spatial network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
"""

import math
import os

import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms


def video_spatial_prediction(
        vid_name,
        net,
        num_categories,
        num_frames=0,
        num_samples=25
):
    if num_frames == 0:
        image_list = os.listdir(vid_name)
        duration = len(image_list)
        # print(duration)
    else:
        duration = num_frames

    clip_mean = [0.485, 0.456, 0.406]
    clip_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=clip_mean, std=clip_std)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # selection
    step = int(math.floor((duration - 1) / (num_samples - 1)))
    dims = (256, 340, 3, num_samples)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num_samples):
        img_file = os.path.join(vid_name, 'image_{0:04d}.jpg'.format(i * step + 1))
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dims[1::-1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb[:, :, :, i] = img
        rgb_flip[:, :, :, i] = img[:, ::-1, :]

    # crop
    rgb_1 = rgb[:224, :224, :, :]
    rgb_2 = rgb[:224, -224:, :, :]
    rgb_3 = rgb[16:240, 60:284, :, :]
    rgb_4 = rgb[-224:, :224, :, :]
    rgb_5 = rgb[-224:, -224:, :, :]
    rgb_f_1 = rgb_flip[:224, :224, :, :]
    rgb_f_2 = rgb_flip[:224, -224:, :, :]
    rgb_f_3 = rgb_flip[16:240, 60:284, :, :]
    rgb_f_4 = rgb_flip[-224:, :224, :, :]
    rgb_f_5 = rgb_flip[-224:, -224:, :, :]

    rgb = np.concatenate((rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_f_1, rgb_f_2, rgb_f_3, rgb_f_4, rgb_f_5), axis=3)

    _, _, _, c = rgb.shape
    rgb_list = []
    for c_index in range(c):
        cur_img = rgb[:, :, :, c_index].squeeze()
        cur_img_tensor = val_transform(cur_img)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))

    rgb_np = np.concatenate(rgb_list, axis=0)
    # print(rgb_np.shape)
    batch_size = 25
    prediction = np.zeros((num_categories, rgb.shape[3]))
    num_batches = int(math.ceil(float(rgb.shape[3]) / batch_size))

    for bb in range(num_batches):
        span = range(batch_size * bb, min(rgb.shape[3], batch_size * (bb + 1)))
        input_data = rgb_np[span, :, :, :]
        img_data_tensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
        img_data_var = torch.autograd.Variable(img_data_tensor)
        output = net(img_data_var)
        result = output.data.cpu().numpy()
        prediction[:, span] = np.transpose(result)

    return prediction
