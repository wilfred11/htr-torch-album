from typing import Callable
import pandas as pd
import csv
import numpy as np
import torch
import torchvision.transforms.functional
from skimage import io
import os
import math
from torchvision.transforms import functional as F, transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import get_size
from files.TextTransform import TextToInt, FillArray
from mystuff.functions import generated_data_dir
import torch.utils.data as data_utils
from matplotlib import pyplot as plt


class FixedHeightResize:
    def __init__(self, size=44):
        self.size = size

    def __call__(self, img):
        sz = get_size(img)
        w = sz[0]
        h = sz[1]
        print('w:', w)
        print('h:', h)

        if h > self.size:
            aspect_ratio = float(h) / float(w)
            new_w = math.ceil(self.size / aspect_ratio)
            return F.resize(img, (self.size, new_w))
        else:
            return img


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

        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
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


class resize_with_padding():
    '''
    Resizes a black and white image to the specified size,
    adding padding to preserve the aspect ratio.
    '''

    def __init__(self, w=128, h=32):
        self.w = w
        self.h = h

    def __call__(self, image):
        # Get the height and width of the image

        sz = get_size(image)
        width = sz[0]
        height = sz[1]

        # Calculate the aspect ratio of the image
        aspect_ratio = height / width

        # Calculate the new height and width after resizing to (224,224)
        new_height = self.h
        new_width = self.w
        if aspect_ratio > 1:
            new_width = int(new_height / aspect_ratio)
        else:
            new_height = int(new_width * aspect_ratio)

        # Resize the image
        resized_image = F.resize(image, (new_width, new_height))

        # Create a black image with the target size
        padded_image = np.zeros((self.w, self.h), dtype=np.uint8)

        # Calculate the number of rows/columns to add as padding
        padding_rows = (self.h - new_height) // 2
        padding_cols = (self.w - new_width) // 2

        if padding_rows > 0 and padding_cols < 0:
            padding_rows = padding_rows // 2
            image = F.pad(image, (0, padding_rows, 0, padding_rows), 0, "constant")
            return F.resize(image, [self.h, self.w])

        elif padding_rows < 0 and padding_cols > 0:
            padding_cols = padding_cols // 2
            image = F.pad(image, (padding_cols, 0, padding_cols, 0), 0, "constant")
            return F.resize(image, [self.h, self.w])

        # Add the resized image to the padded image, with padding on the left and right sides
        # padded_image[padding_rows:padding_rows + new_height, padding_cols:padding_cols + new_width] = resized_image

        # return padded_image


class CustomImageDataset(Dataset):
    def __init__(self, map=None, transform=None, target_transform=None):
        file_names = []
        labels = []
        with open(generated_data_dir() + 'file_names-labels.csv', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            text_to_int = TextToInt(map)
            fill_array = FillArray(length=21)
            for row in reader:
                file_names.append(row[0])
                # labels.append(row[1])
                # print(type(text_to_int(row[1])))
                lbl_tensor = torch.IntTensor(fill_array(text_to_int(row[1])))
                # print(row[1])
                # print(text_to_int(row[1]))
                labels.append(lbl_tensor)

            self.img_labels = labels

            self.img_names = file_names
            self.img_dims = (32, 128)
            self.transform = transform
            self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(self.img_names[idx])
        label = self.img_labels[idx]
        print('label:', label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            print('label transform:', label)
        return image, label


def chars_to_set(word, set):
    for ch in word:
        set.add(ch)
    return set


def all_chars_in_set(str, set):
    for ch in str:
        if ch not in set:
            return False
    return True


def dataset_load(image_transform, char_to_int_map, num_of_rows, text_label_max_length, char_set):
    print('loading dataset')
    images = torch.FloatTensor()
    labels = torch.IntTensor()
    counter = 0
    # torch.set_printoptions(threshold=100000)
    # char_to_int_map = {}
    chars = set()

    with open(generated_data_dir() + 'file_names-labels.csv', newline='') as file:

        reader = csv.reader(file)
        next(reader)
        text_to_int = TextToInt(char_to_int_map)
        fill_array = FillArray(length=text_label_max_length)
        max_height = 0
        max_width = 0
        for row in reader:
            if len(row[1]) > text_label_max_length:
                #print('too long')
                continue
            if not all_chars_in_set(row[1], char_set):
                continue

            chars = chars_to_set(row[1], chars)
            lbl_tensor = torch.IntTensor(fill_array(text_to_int(row[1])))

            img = read_image(row[0])
            #if img.shape[2] > max_width:
            #    plt.imshow(img, cmap='gray')
            #    plt.show()

            img = torchvision.transforms.functional.invert(img)
            image = image_transform(img)

            # print(image)
            if image is None or lbl_tensor is None:
                print('err')
                continue
            #print('img_shp:', img.shape[1])
            if image.shape[1]> max_height:
                max_height= image.shape[1]
            if image.shape[2] > max_width:
                max_width = image.shape[2]

            images = torch.cat((images, image), 0)

            labels = torch.cat((labels, lbl_tensor), 0)
            counter = counter + 1
            if counter == num_of_rows:
                labels = labels.reshape([num_of_rows, text_label_max_length])
                print("len(labels):", str(len(labels)))
                print("len(images):", str(len(images)))
                #print('lblshp:', labels.shape)
                #print('imgshp:', images.shape)
                break

    print("slabels_shp:", labels.shape)
    print("sdata_shp:", images.shape)

    seq_dataset = data_utils.TensorDataset(images, labels)
    train_set, test_set = torch.utils.data.random_split(seq_dataset,
                                                        [int(len(seq_dataset) * 0.8), int(len(seq_dataset) * 0.2)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    print(chars)
    print('max_height:', max_height)
    print('max_width:', max_width)
    return train_loader, test_loader


def dataloader_show(loader):
    # @title Debug: Show Dataset Images

    number_of_printed_imgs = 10

    for batch_id, (x_test, y_test) in enumerate(loader):
        for j in range(len(x_test)):
            plt.imshow(x_test[j], cmap='gray')
            plt.show()

            print(y_test[j])
            number_of_printed_imgs -= 1

            if number_of_printed_imgs <= 0:
                break

        if number_of_printed_imgs <= 0:
            break
