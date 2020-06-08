import cv2
import numpy as np


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, img, scale, mask=None, flip=False):
        img = cv2.resize(img, scale, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) if img.dtype != np.float32 else img.copy()
        if self.to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        cv2.subtract(img, np.float64(self.mean.reshape(1, -1)), img)
        cv2.multiply(img, 1 / np.float64(self.std.reshape(1, -1)), img)

        if flip:
            img = np.flip(img, axis=1)
        img = img.transpose(2, 0, 1)
        
        if mask is not None:
            mask = cv2.resize(mask, scale, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            if flip:
                mask = np.flip(mask, axis=1)
            mask = mask.transpose(2, 0, 1)
            return img, mask
        else:
            return img


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.bd = brightness_delta
        self.cl, self.ch = contrast_range
        self.sl, self.sh = saturation_range
        self.hd = hue_delta

    def __call__(self, img, mask=None, label=None):
        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-self.bd, self.bd)
            img += delta

        if np.random.randint(2):
            alpha = np.random.uniform(self.cl, self.ch)
            img *= alpha

        # convert color from BGR to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(self.sl, self.sh)

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-self.hd, self.hd)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if mask is not None:
            return img, mask, label
        else:
            return img, label


class RandomErasing(object):

    def __init__(self,
                 probability=0.5,
                 area=(0.01, 0.05),
                 mean=(0.4914, 0.4822, 0.4465)):
        self.p = probability
        self.mean = mean
        self.area = area

    def __call__(self, img, mask=None, label=None):
        if np.random.uniform(0, 1) < self.p:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
                target_area = np.random.uniform(self.area[0], self.area[1]) * area
                aspect_ratio = np.random.uniform(0.5, 2)

                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))

                if w < img.shape[1] and h < img.shape[0]:
                    x1 = np.random.randint(0, img.shape[0] - h)
                    y1 = np.random.randint(0, img.shape[1] - w)
                    img[x1:x1 + h, y1:y1 + w, :] = self.mean

        if mask is not None:
            return img, mask, label
        else:
            return img, label


class RandomCutOut(object):
    def __init__(self,
                 probability=0.5,
                 max_edge=50):
        self.p = probability
        self.max_edge = max_edge

    def __call__(self, img, mask=None, label=None):
        if np.random.uniform(0, 1) < self.p:
            h, w = img.shape[0], img.shape[1]
            x, y = np.random.randint(w), np.random.randint(h)
            edge = np.random.randint(1, self.max_edge) // 2

            x1, x2 = np.clip(x - edge, 0, w), np.clip(x + edge, 0, w)
            y1, y2 = np.clip(y - edge, 0, h), np.clip(y + edge, 0, h)
            img[y1:y2, x1:x2, :] = 0

        if mask is not None:
            return img, mask, label
        else:
            return img, label


class RandomRotate(object):
    def __init__(self,
                 probability=0.5,
                 angle=20):
        self.p = probability
        self.angle = angle

    def __call__(self, img, mask=None, label=None):
        if np.random.uniform(0, 1) < self.p:
            h, w = img.shape[0], img.shape[1]
            angle = np.random.randint(-self.angle, self.angle)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h))

        if mask is not None:
            return img, mask, label
        else:
            return img, label


class RandomCrop(object):
    def __init__(self,
                 probability=0.5,
                 w_h=(0.1, 0.1)):
        self.p = probability
        self.w_h = w_h

    def __call__(self, img, mask=None, label=None):
        if np.random.uniform(0, 1) < self.p:
            h, w = img.shape[0], img.shape[1]
            _w, _h = self.w_h
            x1 = np.random.randint(0, int(_w * w))
            y1 = np.random.randint(0, int(_h * h))
            x2 = np.random.randint(int((1-_w) * w), w)
            y2 = np.random.randint(int((1-_h) * h), h)
            img = img[y1:y2, x1:x2, :]
            if mask is not None:
                mask = mask[y1:y2, x1:x2, :]

        if mask is not None:
            return img, mask, label
        else:
            return img, label


class RandomPatch(object):
    def __init__(self,
                 range_w=(0.3, 0.8),
                 range_h=(0.2, 0.7),
                 size=112):
        self.range_w = range_w
        self.range_h = range_h
        self.size = size

    def __call__(self, img, mask=None, label=None):
        h, w = img.shape[0], img.shape[1]
        xmin, ymin = int(w * self.range_w[0]), int(h * self.range_h[0])
        xmax, ymax = int(w * self.range_w[1]), int(h * self.range_h[1])
        x_start = np.random.randint(xmin, xmax)
        y_start = np.random.randint(ymin, ymax)
        x1 = max(x_start - self.size, 0)
        y1 = max(y_start - self.size, 0)
        x2 = min(x_start + self.size, w)
        y2 = min(y_start + self.size, h)
        img = img[y1:y2, x1:x2, :]
        if mask is not None:
            mask = mask[y1:y2, x1:x2, :]
            return img, mask, label
        else:
            return img, label


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 random_erasing=None,
                 random_cutout=None,
                 ramdom_rotate=None,
                 ramdom_crop=None,
                 ramdom_patch=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if random_erasing is not None:
            self.transforms.append(
                RandomErasing(**random_erasing))
        if random_cutout is not None:
            self.transforms.append(
                RandomCutOut(**random_cutout))
        if ramdom_rotate is not None:
            self.transforms.append(
                RandomRotate(**ramdom_rotate))
        if ramdom_crop is not None:
            self.transforms.append(
                RandomCrop(**ramdom_crop))
        if ramdom_patch is not None:
            self.transforms.append(
                RandomPatch(**ramdom_patch))

    def __call__(self, img, mask=None, label=None):
        for tranform in self.transforms:
            if mask is not None:
                img, mask, label = tranform(img, mask, label)
            else:
                img, label = tranform(img, label=label)

        if mask is not None:
            return img, mask, label
        else:
            return img, label

