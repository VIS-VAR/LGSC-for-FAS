import os
import cv2
import paddle.fluid as fluid
import numpy as np
from .transforms import ImageTransform, ExtraAugmentation


class DatasetBase(object):
    """
    Base Dataset config
    """
    def __init__(self,
                 img_prefix,
                 ann_file,
                 img_scale,
                 img_norm_cfg,
                 extra_aug=None,
                 test_mode=False):
        self.img_prefix = img_prefix
        self.img_scale = img_scale
        self.extra_aug = extra_aug
        self.test_mode = test_mode
        self.img_norm_cfg = img_norm_cfg
        self.img_infos = self.load_annotations(ann_file)
        self.img_transform = ImageTransform(**img_norm_cfg)
        self.extra_aug = ExtraAugmentation(**extra_aug)

    def load_annotations(self, ann_file):
        img_infos = []
        with open(ann_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                item = line.strip().split(' ')
                img_infos.append(dict(
                    img_path=item[0],
                    label=int(item[1])
                ))
        return img_infos

    def length(self):
        return len(self.img_infos)

    def train(self, batch_size=None):

        def reader():
            batch = []
            for img_info in self.img_infos:
                img_path = img_info['img_path']
                label = img_info['label']
                img = cv2.imread(img_path)
                img, label = self.extra_aug(img, label=label)
                flip = True if np.random.rand() < 0.5 else False
                img = self.img_transform(img, self.img_scale, flip=flip)

                if batch_size is None:
                    yield img, label
                else:
                    batch.append([img, label])
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
        return reader

    def test(self):
        def reader():
            for img_info in self.img_infos:
                img_path = img_info['img_path']
                label = img_info['label']
                img = cv2.imread(img_path)
                img = self.img_transform(img, self.img_scale)
                yield img, label
        return reader

    def data_loader(self, batch_size, place):

        def _loader():
            data_loader = fluid.io.DataLoader.from_generator(capacity=128, use_multiprocess=True, iterable=True)
            data_loader.set_sample_list_generator(self.train(batch_size=batch_size), places=place)
            batch = []
            for data in data_loader():
                imgs, masks, labels = data[0].numpy(), data[1].numpy(), data[2].numpy()
                for i in range(batch_size):
                    batch.append([imgs[i], masks[i], labels[i]])
                yield batch
                batch = []

        return _loader



