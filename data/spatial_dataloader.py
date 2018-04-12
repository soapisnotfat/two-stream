from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import pickle

from .split_train_test_video import *


class SpatialDataset(Dataset):
    def __init__(self, dic, root_dir, mode, transform=None):

        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self, video_name, index):
        path = self.root_dir + 'v_' + video_name + '/frame'

        img = Image.open(path + str(index).zfill(6) + '.jpg')
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = [random.randint(1, nb_clips // 3), random.randint(nb_clips // 3, nb_clips * 2 // 3),
                     random.randint(nb_clips * 2 // 3, nb_clips + 1)]

        elif self.mode == 'val':
            video_name, index = self.keys[idx].split(' ')
            index = abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label) - 1

        if self.mode == 'train':
            data = {}
            for i in range(len(clips)):
                key = 'img' + str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index)

            sample = (data, label)
        elif self.mode == 'val':
            data = self.load_ucf_image(video_name, index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')

        return sample


class SpatialDataloader(object):
    def __init__(self, batch_size, num_workers, path, ucf_list, ucf_split):

        self.BATCH_SIZE = batch_size
        self.num_workers = num_workers
        self.data_path = path
        self.frame_count = {}

        # split the training and testing videos
        _splitter = UCF101Splitter(path=ucf_list, split=ucf_split)
        self.train_video, self.test_video = _splitter.split_video()

        self.dic_training = dict()
        self.dic_testing = dict()

    def load_frame_count(self):
        # print '==> Loading frame number of each video'
        with open('data/dic/frame_count.pickle', 'rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame:
            video_name = line.split('_', 1)[1].split('.', 1)[0]
            self.frame_count[video_name] = dic_frame[line]

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample20()
        training_loader = self.train()
        validate_loader = self.validate()

        return training_loader, validate_loader, self.test_video

    def get_training_dic(self):
        # print '==> Generate frame numbers of each training video'
        self.dic_training = dict()

        for video in self.train_video:
            # print video_name
            nb_frame = self.frame_count[video] - 10 + 1
            key = video + ' ' + str(nb_frame)
            self.dic_training[key] = self.train_video[video]

    def val_sample20(self):
        print('==> sampling testing frames')
        self.dic_testing = dict()

        for video in self.test_video:
            nb_frame = self.frame_count[video] - 10 + 1
            interval = int(nb_frame / 19)

            for i in range(19):
                frame = i * interval
                key = video + ' ' + str(frame + 1)
                self.dic_testing[key] = self.test_video[video]

    def train(self):
        training_set = SpatialDataset(dic=self.dic_training, root_dir=self.data_path, mode='train',
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ]))

        print('==> Training data :', len(training_set), 'frames')

        training_loader = DataLoader(dataset=training_set, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.num_workers)
        return training_loader

    def validate(self):
        validation_set = SpatialDataset(dic=self.dic_testing, root_dir=self.data_path, mode='val',
                                        transform=transforms.Compose([
                                            transforms.Resize([224, 224]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ]))

        print('==> Validation data :', len(validation_set), 'frames')

        validate_loader = DataLoader(dataset=validation_set, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.num_workers)
        return validate_loader


if __name__ == '__main__':
    data_loader = SpatialDataloader(batch_size=1, num_workers=1,
                                    path='../UCF101/jpegs_256/',
                                    ucf_list='../UCF101/UCF_list/',
                                    ucf_split='01')
    train_loader, val_loader, test_video = data_loader.run()
