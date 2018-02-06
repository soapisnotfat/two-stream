import math
import os
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

import models
from evaluate.video_spatial_prediction import video_spatial_prediction

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k / sum_y for k in y]

    return z


def main():
    model_path = '../../checkpoints/model_best.pth.tar'
    start_frame = 0
    num_categories = 101

    model_start_time = time.time()
    params = torch.load(model_path)

    spatial_net = models.rgb_resnet152(pretrained=False, num_classes=101)
    spatial_net.load_state_dict(params['state_dict'])
    spatial_net.cuda()
    spatial_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % model_time)

    val_file = "./testlist01_with_labels.txt"
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count = 0
    result_list = []
    for line in val_list:
        line_info = line.split(" ")
        clip_path = line_info[0]
        input_video_label = int(line_info[1]) - 1

        spatial_prediction = video_spatial_prediction(
            clip_path,
            spatial_net,
            num_categories)

        avg_spatial_prediction_fc8 = np.mean(spatial_prediction, axis=1)
        # print(avg_spatial_prediction_fc8.shape)
        result_list.append(avg_spatial_prediction_fc8)
        # avg_spatial_pred = softmax(avg_spatial_prediction_fc8)

        prediction_index = np.argmax(avg_spatial_prediction_fc8)
        print("Sample %d/%d: GT: %d, Prediction: %d" % (line_id, len(val_list), input_video_label, int(prediction_index)))

        if prediction_index == input_video_label:
            match_count += 1
        line_id += 1

    print(match_count)
    print(len(val_list))
    print("Accuracy is %4.4f" % (float(match_count) / len(val_list)))
    np.save("ucf101_s1_rgb_resnet152.npy", np.array(result_list))


if __name__ == "__main__":
    main()
