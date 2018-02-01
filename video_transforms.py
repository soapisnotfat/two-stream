from __future__ import division
import random
import numpy as np
import cv2


class MultiScaleCrop(object):
    """
    Description: Corner cropping and multi-scale cropping. Two data augmentation techniques introduced in:
        Towards Good Practices for Very Deep Two-Stream ConvNets,
        http://arxiv.org/abs/1507.02159
        Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao

    Parameters:
        size: height and width required by network input, e.g., (224, 224)
        scale_ratios: efficient scale jittering, e.g., [1.0, 0.875, 0.75, 0.66]
        fix_crop: use corner cropping or not. Default: True
        more_fix_crop: use more corners or not. Default: True
        max_distort: maximum distortion. Default: 1
        interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, size, scale_ratios, fix_crop=True, more_fix_crop=True, max_distort=1,
                 interpolation=cv2.INTER_LINEAR):
        self.height = size[0]
        self.width = size[1]
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.interpolation = interpolation

    def fill_fix_offset(self, datum_height, datum_width):
        h_off = int((datum_height - self.height) / 4)
        w_off = int((datum_width - self.width) / 4)

        offsets = list()
        offsets.append((0, 0))                      # upper left
        offsets.append((0, 4 * w_off))              # upper right
        offsets.append((4 * h_off, 0))              # lower left
        offsets.append((4 * h_off, 4 * w_off))      # lower right
        offsets.append((2 * h_off, 2 * w_off))      # center

        if self.more_fix_crop:
            offsets.append((0, 2 * w_off))          # top center
            offsets.append((4 * h_off, 2 * w_off))  # bottom center
            offsets.append((2 * h_off, 0))          # left center
            offsets.append((2 * h_off, 4 * w_off))  # right center

            offsets.append((1 * h_off, 1 * w_off))  # upper left quarter
            offsets.append((1 * h_off, 3 * w_off))  # upper right quarter
            offsets.append((3 * h_off, 1 * w_off))  # lower left quarter
            offsets.append((3 * h_off, 3 * w_off))  # lower right quarter

        return offsets

    def fill_crop_size(self, input_height, input_width):
        crop_sizes = []
        base_size = np.min((input_height, input_width))
        scale_rates = self.scale_ratios
        for h in range(len(scale_rates)):
            crop_h = int(base_size * scale_rates[h])
            for w in range(len(scale_rates)):
                crop_w = int(base_size * scale_rates[w])
                # append this cropping size into the list
                if np.absolute(h - w) <= self.max_distort:
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def __call__(self, clips):
        height, w, c = clips.shape
        is_color = False
        if c % 3 == 0:
            is_color = True

        crop_size_pairs = self.fill_crop_size(height, w)
        size_sel = random.randint(0, len(crop_size_pairs) - 1)
        crop_height = crop_size_pairs[size_sel][0]
        crop_width = crop_size_pairs[size_sel][1]

        if self.fix_crop:
            offsets = self.fill_fix_offset(height, w)
            off_sel = random.randint(0, len(offsets) - 1)
            h_off = offsets[off_sel][0]
            w_off = offsets[off_sel][1]
        else:
            h_off = random.randint(0, height - self.height)
            w_off = random.randint(0, w - self.width)

        scaled_clips = np.zeros((self.height, self.width, c))
        if is_color:
            num_images = int(c / 3)
            for frame_id in range(num_images):
                cur_img = clips[:, :, frame_id * 3:frame_id * 3 + 3]
                crop_img = cur_img[h_off:h_off + crop_height, w_off:w_off + crop_width, :]
                scaled_clips[:, :, frame_id * 3:frame_id * 3 + 3] = cv2.resize(crop_img, (self.width, self.height),
                                                                               self.interpolation)
            return scaled_clips
        else:
            num_images = int(c / 1)
            for frame_id in range(num_images):
                cur_img = clips[:, :, frame_id:frame_id + 1]
                crop_img = cur_img[h_off:h_off + crop_height, w_off:w_off + crop_width, :]
                scaled_clips[:, :, frame_id:frame_id + 1] = np.expand_dims(
                    cv2.resize(crop_img, (self.width, self.height), self.interpolation), axis=2)
            return scaled_clips
