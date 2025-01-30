import cv2
from albumentations import ImageOnlyTransform
from torchvision.transforms import functional as F
from torchvision.transforms.v2.functional import get_size
import albumentations as A


class TextToInt:

    def __init__(self, char_to_int_map):
        self.char_map = char_to_int_map

    def __call__(self, text):
        int_sequence = []
        for c in text:
            if c == " ":
                ch = self.char_map[""]
            else:
                ch = self.char_map[c]
            int_sequence.append(int(ch))
        return int_sequence


class IntToText:
    def __init__(self, int_to_char_map):
        self.int_map = int_to_char_map

    def __call__(self, integer_tensor):
        char_sequence = []
        for i in integer_tensor:
            if i == " ":
                ch = self.int_map[""]
            else:
                ch = self.int_map[str(i.item())]
            char_sequence.append(ch)
        return char_sequence


class IntToString:
    def __init__(self, int_to_char_map):
        self.int_map = int_to_char_map

    def __call__(self, integer_tensor):
        char_sequence = []
        for i in integer_tensor:
            if i == " ":
                ch = self.int_map[""]
            else:
                ch = self.int_map[str(i.item())]
            char_sequence.append(ch)
        string = "".join([str(c) for c in char_sequence])
        return string


class FillArray:

    def __init__(self, length, empty_label):
        self.length = length
        self.empty_label = empty_label

    def __call__(self, array):
        for i in range(self.length - len(array)):
            array.append(self.empty_label)

        return array


class ResizeWithPad:
    def __init__(self, w=156, h=44):
        self.w = w
        self.h = h

    def __call__(self, image):
        sz = get_size(image)
        w_1 = sz[0]
        h_1 = sz[1]

        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1

        if round(ratio_1, 2) != round(ratio_f, 2):
            hp = int(w_1 / ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])

        else:
            return F.resize(image, [self.h, self.w])


class AResizeWithPad(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=1, w=156, h=44):
        super().__init__(always_apply, p)
        self.w = w
        self.h = h

    def apply(self, image, **params):
        # print(image.shape)
        sz = (image.shape[2], image.shape[1])
        # sz = get_size(image)
        w_1 = sz[0]
        h_1 = sz[1]

        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1

        if round(ratio_1, 2) != round(ratio_f, 2):
            hp = int(w_1 / ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])

        else:
            return F.resize(image, [self.h, self.w])


"""def replay_transform():
    return A.ReplayCompose(
        [
            '''A.OneOf(
                [
                    A.GaussNoise(p=1),
                    A.Blur(p=1),
                    A.RandomGamma(p=1),
                    #A.GridDistortion(p=1),
                    # A.PixelDropout(p=1, drop_value=None),
                    A.Morphological(p=1, scale=(4, 6), operation="dilation"),
                    A.Morphological(p=1, scale=(4, 6), operation="erosion"),
                    A.RandomBrightnessContrast(p=1),

                ],
                p=1,
            )''',
            A.Affine(p=1, rotate=5)
            #A.SafeRotate(limit=(-15.75, 15.75), p=1)
            # A.InvertImg(p=1),
            # AResizeWithPad(h=44, w=156),
        ]
    )"""

def replay_transform():
    return A.ReplayCompose(
        [
            A.Rotate(limit=(-4.5,4.5), p=1.0, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf(
                [
                    A.GaussNoise(p=1, var_limit=(233,255), per_channel=False),
                    A.Blur(p=1, blur_limit=3),
                    #A.GridDistortion(p=1, distort_limit=(-0.3,0.3),normalized=True, interpolation=cv2.INTER_NEAREST),
                    A.Morphological(p=1, scale=(3, 4), operation="dilation"),
                    A.Morphological(p=1, scale=(3, 4), operation="erosion"),
                    A.PixelDropout(p=1, drop_value=133, dropout_prob=.04)
                ],
                p=1.0,
            )
        ]
    )

def train_transform():
    return A.Compose(
        [
            A.Rotate(limit=(-4.5, 4.5), p=0.25, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf(
            [

                A.GaussNoise(p=1, var_limit=(233, 255), per_channel=False),
                A.Blur(p=1, blur_limit=3),
                A.Morphological(p=1, scale=(3, 4), operation="dilation"),
                A.Morphological(p=1, scale=(3, 4), operation="erosion", ),
                A.PixelDropout(p=1, drop_value=133, dropout_prob=.04)
            ],
            p=0.25,
            )
        ]
    )

def pretrain_transform():
    return A.Compose(
        [
                A.Morphological(p=1, scale=(2, 3), operation="dilation"),
                A.RandomBrightnessContrast(brightness_limit=[1,1], contrast_limit = [1,1],brightness_by_max=False, p = 1.0),
                #A.ColorJitter(brightness=0, contrast=(1, 5), saturation=0, hue=0)
                A.Rotate(limit=(-4.5, 4.5), p=0.25, border_mode=cv2.BORDER_CONSTANT),
                A.OneOf(
                    [

                        A.GaussNoise(p=1, var_limit=(233, 255), per_channel=False),
                        A.Blur(p=1, blur_limit=3),
                        A.Morphological(p=1, scale=(3, 4), operation="dilation"),
                        A.Morphological(p=1, scale=(3, 4), operation="erosion", ),
                        A.PixelDropout(p=1, drop_value=133, dropout_prob=.04)
                    ],
                    p=0.25,
                )
        ],
        p=1,
        )
"""A.OneOf(
                [

                    #A.GaussNoise(p=1),
                    #A.Blur(p=1),
                    #A.RandomGamma(p=1),
                    #A.GridDistortion(p=1),
                    # A.PixelDropout(p=1, drop_value=None),
                    #A.Morphological(p=1, scale=(4, 6), operation="dilation"),
                    #A.Morphological(p=1, scale=(4, 6), operation="erosion"),
                    #A.RandomBrightnessContrast(p=1),
                    #A.Affine(p=1),
                ],
                p=0.25,
            ),"""