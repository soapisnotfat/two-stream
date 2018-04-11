import pickle
import numpy as np

from misc import *
import dataloader


if __name__ == '__main__':

    rgb_predictions = 'record/spatial/spatial_video_preds.pickle'
    optic_flow_predictions = 'record/motion/motion_video_preds.pickle'

    with open(rgb_predictions, 'rb') as f:
        rgb = pickle.load(f)
    f.close()
    with open(optic_flow_predictions, 'rb') as f:
        opf = pickle.load(f)
    f.close()

    dataloader = dataloader.SpatialDataloader(batch_size=1, num_workers=1,
                                              path='./UCF101/jpegs_256/',  # TODO ???
                                              ucf_list='./UCF_list/',
                                              ucf_split='01')
    train_loader, val_loader, test_video = dataloader.run()

    video_level_predictions = np.zeros((len(rgb.keys()), 101))
    video_level_labels = np.zeros(len(rgb.keys()))
    correct = 0
    ii = 0
    for name in sorted(rgb.keys()):
        r = rgb[name]
        o = opf[name]

        label = int(test_video[name]) - 1

        video_level_predictions[ii, :] = (r + o)
        video_level_labels[ii] = label
        ii += 1
        if np.argmax(r + o) == label:
            correct += 1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_predictions = torch.from_numpy(video_level_predictions).float()

    top1, top5 = accuracy(video_level_predictions, video_level_labels, topk=(1, 5))

    print(top1, top5)
