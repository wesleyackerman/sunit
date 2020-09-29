import numbers
from PIL import Image
import torchvision.transforms.functional as TF
import random

class PairCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img1, img2):
        assert img1.size == img2.size
        for tf in self.transforms:
            img1, img2 = tf(img1, img2)
        return img1, img2

class PairRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def random_crop_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw) 
        return i, j, th, tw

    def __call__(self, img1, img2):
        i, j, h, w = self.random_crop_params(img1, self.size)
        return TF.crop(img1, i, j, h, w), TF.crop(img2, i, j, h, w)

class PairResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img1, img2):
        return TF.resize(img1, self.size, self.interpolation), TF.resize(img2, self.size, self.interpolation)

class PairRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            return TF.hflip(img1), TF.hflip(img2)
        return img1, img2

