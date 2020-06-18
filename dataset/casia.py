import os
import cv2
import numpy as np
import pickle
from .datasetbase import DatasetBase


class CASIA(DatasetBase):

    def load_annotations(self, ann_file):

        with open(ann_file, 'rb') as pkl:
            img_info_dict = pickle.load(pkl)
        img_infos = []
        if self.test_mode:
            for k, v in img_info_dict.items():
                v['filename'] = k
                img_infos.append(v)
        else:
            for k, v in img_info_dict.items():
                for frame in v['frames'][0:]:
                    img_infos.append(dict(
                        filename=k,
                        frames=[frame],
                        labels=[v['labels'][0]]))
        print('total number of data:', len(img_infos))
        return img_infos

    def _get_mask(self, img, thr=10, crop=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, thr, 1, cv2.THRESH_BINARY)
        indy, indx = np.where(mask > 0)
        x1, y1 = indx.min(), indy.min()
        x2, y2 = indx.max(), indy.max()
        y1 = int((y1+y2)/2)
        if crop:
            return img[y1:y2, x1:x2, :], mask[y1:y2, x1:x2]
        else:
            return img, mask

    def train(self, batch_size=None):
        np.random.shuffle(self.img_infos)

        def reader():
            batch = []
            for img_info in self.img_infos:
                img_path = os.path.join(self.img_prefix, img_info['filename'],
                                        'profile', img_info['frames'][0])
                label = img_info['labels'][0]
                img = cv2.imread(img_path)
                img = img.astype(np.float32)
                mask = np.zeros_like(img)
                
                img, mask, label = self.extra_aug(img, mask=mask, label=label)

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
                img_path = os.path.join(self.img_prefix, img_info['filename'],
                                        'profile', img_info['frames'][-1])
                label = img_info['labels'][-1]
                img = cv2.imread(img_path)
                img = self.img_transform(img, self.img_scale)
                yield img, label

        return reader



