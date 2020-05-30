import os
import cv2
import numpy as np
from .datasetbase import DatasetBase


class FaceForensics(DatasetBase):

    def __init__(self,
                 img_prefix,
                 ann_file,
                 img_scale,
                 img_norm_cfg,
                 extra_aug=None,
                 test_mode=False,
                 mask_file=None,
                 crop_face=0):
        self.with_mask = mask_file is not None
        self.mask_file = mask_file
        self.crop_face = crop_face
        super(FaceForensics, self).__init__(
            img_prefix,
            ann_file,
            img_scale,
            img_norm_cfg,
            extra_aug,
            test_mode)

    def load_annotations(self, ann_file):

        pos_infos = []
        neg_infos = []
        img_infos = []
        self.kp_dict = dict()

        if not self.test_mode and self.with_mask:
            with open(self.mask_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    data = line.strip().split(' ')
                    kp = data[1:]
                    kps = [(int(kp[i]), int(kp[i + 1])) for i in range(0, len(kp), 2)]
                    self.kp_dict[data[0]] = kps

        with open(ann_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                item = line.strip().split(' ')
                if item[0] not in self.kp_dict and self.with_mask and not self.test_mode:
                    continue
                if self.test_mode:
                    img_infos.append(dict(
                        img_path=item[0],
                        label=1 - int(item[1])))
                if int(item[1]) == 0:
                    pos_infos.append(dict(
                        img_path=item[0],
                        label=1 - int(item[1])))
                else:
                    neg_infos.append(dict(
                        img_path=item[0],
                        label=1 - int(item[1])))
        pos_len = len(pos_infos)
        neg_len = len(neg_infos)

        print('total number of data: {} | pos: {}, neg: {}'.format(
            pos_len + neg_len, pos_len, neg_len))

        if self.test_mode:
            return img_infos
        else:
            if pos_len > neg_len:
                neg_infos = neg_infos * int(float(pos_len) / neg_len + 1)
                neg_infos = neg_infos[:pos_len]
            else:
                pos_infos = pos_infos * int(float(neg_len) / pos_len + 1)
                pos_infos = pos_infos[:neg_len]

            img_infos = pos_infos + neg_infos
            print('After balance total number of data: {} | pos: {}, neg: {}'.format(
                len(img_infos), len(pos_infos), len(neg_infos)))

            return img_infos

    def _get_mask(self, kps, img):

        def expand_eyebrows(lmrks, eyebrows_expand_mod=2.0):
            bot_l = np.array(lmrks[13:18])
            bot_r = np.array(lmrks[30:35])
            top_l = np.array(lmrks[22:27])
            top_r = np.array(lmrks[39:44])

            top_l = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
            top_r = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)

            lmrks[22:27] = [tuple(top_l[i, :].astype(int)) for i in range(5)]
            lmrks[39:44] = [tuple(top_r[i, :].astype(int)) for i in range(5)]

            return lmrks

        kps = expand_eyebrows(kps)
        poly = kps[0:1] + kps[22:26] + kps[40:43] + kps[0:13][::-1]
        mask = np.zeros_like(img, dtype=np.uint8)
        mask = cv2.fillPoly(mask, [np.array(poly)], (1, 1, 1))
        return mask

    def _get_face(self, img, mask=None, thr=0):
        assert thr <= 0.2
        h, w, _ = img.shape
        x1, x2 = int(thr * w), int((1 - thr) * w)
        y1, y2 = 0, int((1 - 1.5 * thr) * h)
        img = img[y1:y2, x1:x2, :]
        if mask is None:
            return img
        mask = mask[y1:y2, x1:x2]
        return img, mask

    def train(self, batch_size=None):
        pos_infos = self.img_infos[:int(len(self.img_infos) / 2)]
        neg_infos = self.img_infos[int(len(self.img_infos) / 2):]
        assert len(pos_infos) == len(neg_infos)

        def reader():
            np.random.shuffle(pos_infos)
            np.random.shuffle(neg_infos)
            img_infos = []

            for i in range(len(pos_infos)):
                img_infos.append(pos_infos[i])
                img_infos.append(neg_infos[i])

            batch = []
            for img_info in img_infos:
                img_path = os.path.join(self.img_prefix,
                                        *[p for p in img_info['img_path'].split('/')[-7:]])
                label = img_info['label']
                img = cv2.imread(img_path)

                mask = self._get_mask(self.kp_dict[img_info['img_path']],
                                      img) if self.with_mask else np.zeros_like(img)

                if self.crop_face:
                    img, mask = self._get_face(img, mask, thr=self.crop_face)

                img = img.astype(np.float32)
                img, mask, label = self.extra_aug(img, mask=mask, label=label)

                flip = True if np.random.rand() < 0.5 else False
                img, mask = self.img_transform(img, self.img_scale, mask=mask, flip=flip)

                if batch_size is None:
                    yield img, mask, label
                else:
                    batch.append([img, mask, label])
                    if len(batch) == batch_size:
                        yield batch
                        batch = []

        return reader

    def test(self):
        def reader():
            for img_info in self.img_infos:
                img_path = os.path.join(self.img_prefix,
                                        *[p for p in img_info['img_path'].split('/')[-2:]])
                label = img_info['label']
                img = cv2.imread(img_path)

                if self.crop_face:
                    img = self._get_face(img, thr=self.crop_face)
                # img = np.flip(img, axis=1)
                img = self.img_transform(img, self.img_scale)
                yield img, label

        return reader

