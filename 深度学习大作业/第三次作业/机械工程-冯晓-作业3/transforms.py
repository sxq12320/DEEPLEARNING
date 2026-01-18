import numpy as np
import random
from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor:
    def __call__(self, img):
        img = np.array(img).astype(np.float32) / 255.
        img = np.transpose(img, (2, 0, 1))
        return img

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img - self.mean) / self.std
        return img

class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                img = img.resize(self.size, Image.BILINEAR)
                return img

        scale = (self.size[0]/img.size[0], self.size[1]/img.size[1])
        w = img.size[0] * min(scale)
        h = img.size[1] * min(scale)
        x1 = (img.size[0] - w) / 2
        y1 = (img.size[1] - h) / 2
        img = img.crop((x1, y1, x1 + w, y1 + h))
        img = img.resize(self.size, Image.BILINEAR)
        return img

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size, self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        return img

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img.resize((self.size, self.size), Image.BILINEAR)
        return img