import csv
import sys

import numpy as np
import torchvision
from torch import IntTensor
from torch.nn import functional as F1
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from files.functions import htr_ds_dir, generated_data_dir
import torch

from files.transform import TextToInt, FillArray, IntToText


def pad_image_to_nearest_multiple(image, multiple=256):
    channels, height, width = image.shape
    padded_height = ((height + multiple - 1) // multiple) * multiple
    padded_width = ((width + multiple - 1) // multiple) * multiple

    pad_bottom = padded_height - height
    pad_right = padded_width - width

    padded_image = F1.pad(image, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
    return padded_image


class CustomObjectDetectionDataset(Dataset):
    def __init__(self, annotations_file, image_folder, number):
        self.labels = torch.FloatTensor()
        self.image = torch.FloatTensor()
        self.annotations_file = annotations_file
        self.image_folder = image_folder
        self.file_names = dict()

        with open(self.annotations_file, newline='') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)
            counter = 5
            current_file_name = ''
            for row in reader:
                if row[1] == '827' and row[2] == '1170':
                    if not row[0] == current_file_name or current_file_name == '':
                        self.file_names[counter] = row[0]
                        current_file_name = row[0]
                        counter = counter + 1
                if counter == number:
                    break

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        with (open(self.annotations_file, newline='') as file):
            reader = csv.reader(file, delimiter=',')
            next(reader)
            lbl_ = []
            found = False
            for row in reader:
                if file_name == row[0]:
                    found = True
                    lbl = torch.FloatTensor(
                        [float(0), float(row[4]), float(row[5]), float(row[6]), float(row[7])]).unsqueeze(0)
                    lbl_lst = [[float(0), float(row[4]), float(row[5]), float(row[6]), float(row[7])]]
                    lbl_.append(lbl_lst)
                elif found and not row[0] == file_name:
                    t_lbls = torch.from_numpy(np.array(lbl_, dtype=np.float32)).squeeze(0).squeeze(0)
                    self.labels = t_lbls
                    img = torch.FloatTensor()
                    img = read_image(htr_ds_dir() + 'train/' + file_name)
                    im_padded = pad_image_to_nearest_multiple(img, 256)
                    transform_norm = v2.Compose([
                        v2.ToDtype(torch.float32, scale=False),
                    ])
                    img = transform_norm(im_padded)
                    self.image = img

        return self.image, self.labels


def all_chars_in_set(str, set):
    for ch in str:
        if ch not in set:
            return False
    return True


class HTRDataset(Dataset):
    def __init__(self, file_name, text_label_max_length, char_to_int_map, int_to_char_map, char_set, image_transform,
                 num_of_rows):
        self.labels = torch.IntTensor()
        self.images = torch.FloatTensor()
        counter = 0
        with open(generated_data_dir() + file_name, newline='') as file:
            reader = csv.reader(file)
            next(reader)
            text_to_int = TextToInt(char_to_int_map)
            fill_array = FillArray(length=text_label_max_length)
            for row in reader:
                if len(row[1]) > text_label_max_length:
                    continue
                if not all_chars_in_set(row[1], char_set):
                    continue

                lbl_tensor = torch.IntTensor(fill_array(text_to_int(row[1])))
                img = read_image(row[0])
                img = torchvision.transforms.functional.invert(img)
                image = image_transform(img)

                if image is None or lbl_tensor is None:
                    continue

                self.images = torch.cat((self.images, image), 0)
                self.labels = torch.cat((self.labels, lbl_tensor), 0)
                counter = counter + 1
                if counter == num_of_rows:
                    self.labels = self.labels.reshape([num_of_rows, text_label_max_length])
                    break
        print('size images:',sys.getsizeof(self.images))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
