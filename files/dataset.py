import csv
import os
import pickle
import random
import sys

import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch import tensor
from collections import Counter
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from files.functions import htr_ds_dir, generated_data_dir, external_data_dir
from sklearn.model_selection import KFold
import albumentations as A
import torch
from files.transform import (
    TextToInt,
    FillArray,
    ResizeWithPad,
)
from pathlib import Path
import torch.nn.functional as F1

def pad_image_to_nearest_multiple(image, multiple=256):
    channels, height, width = image.shape
    padded_height = ((height + multiple - 1) // multiple) * multiple
    padded_width = ((width + multiple - 1) // multiple) * multiple

    pad_bottom = padded_height - height
    pad_right = padded_width - width

    padded_image = F1.pad(
        image, (0, pad_right, 0, pad_bottom), mode="constant", value=0
    )
    return padded_image


class CustomObjectDetectionDataset(Dataset):
    def __init__(self, annotations_file, image_folder, number):
        self.labels = torch.FloatTensor()
        self.image = torch.FloatTensor()
        self.annotations_file = annotations_file
        self.image_folder = image_folder
        self.file_names = dict()

        with open(self.annotations_file, newline="") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            counter = 5
            current_file_name = ""
            for row in reader:
                if row[1] == "827" and row[2] == "1170":
                    if not row[0] == current_file_name or current_file_name == "":
                        self.file_names[counter] = row[0]
                        current_file_name = row[0]
                        counter = counter + 1
                if counter == number:
                    break

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        with open(self.annotations_file, newline="") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            lbl_ = []
            found = False
            for row in reader:
                if file_name == row[0]:
                    found = True
                    lbl = torch.FloatTensor(
                        [
                            float(0),
                            float(row[4]),
                            float(row[5]),
                            float(row[6]),
                            float(row[7]),
                        ]
                    ).unsqueeze(0)
                    lbl_lst = [
                        [
                            float(0),
                            float(row[4]),
                            float(row[5]),
                            float(row[6]),
                            float(row[7]),
                        ]
                    ]
                    lbl_.append(lbl_lst)
                elif found and not row[0] == file_name:
                    t_lbls = (
                        torch.from_numpy(np.array(lbl_, dtype=np.float32))
                        .squeeze(0)
                        .squeeze(0)
                    )
                    self.labels = t_lbls
                    img = torch.FloatTensor()
                    img = read_image(htr_ds_dir() + "train/" + file_name)
                    im_padded = pad_image_to_nearest_multiple(img, 256)
                    transform_norm = v2.Compose(
                        [
                            v2.ToDtype(torch.float32, scale=False),
                        ]
                    )
                    img = transform_norm(im_padded)
                    self.image = img

        return self.image, self.labels


def all_chars_in_set(str, set):
    for ch in str:
        if ch not in set:
            return False
    return True


class AHTRDatasetOther(Dataset):
    def __init__(
        self,
        file_name,
        config,
        image_transform,
        num_of_rows,
    ):
        self.labels = torch.IntTensor()
        self.img_names = []
        self.label_lengths = []
        list_of_images = []
        counter = 0
        with open(external_data_dir()+'/handwriting-generation/' + file_name,'rb') as f:
            dirs = pickle.load(f)
            text_to_int = TextToInt(config.char_to_int_map)
            fill_array = FillArray(
                length=config.text_label_max_length, empty_label=config.empty_label
            )
            #print(dirs)
            dirs = list(set(dirs))
            dirs.sort()
            for d in dirs:
                d_split = d.split("/")
                length = len(d_split)
                if length==6:
                    #print(d)
                    print(d_split[2])
                    if d_split[2]== '0':
                        print(d_split[3])
                        if d_split[3] in config.char_set:
                            c_dir = external_data_dir()+'handwriting-generation/'+d
                            print(c_dir)
                            print(os.listdir(c_dir))
                            for f in os.listdir(c_dir):
                                lbl=Path(f).stem
                                lbl_tensor = torch.IntTensor(fill_array(text_to_int(lbl)))

                                img = read_image(c_dir + f,'RGB')
                                img = torchvision.transforms.functional.adjust_contrast(img, contrast_factor=9000000000)
                                img = torchvision.transforms.functional.invert(img)
                                t = ResizeWithPad(w=156, h=44)
                                img = t(img)

                                if img is None or lbl_tensor is None:
                                    continue

                                list_of_images.append(img)
                                self.img_names.append(f)
                                self.labels = torch.cat((self.labels, lbl_tensor), 0)
                                self.label_lengths.append(len(lbl))

                                counter = counter + 1
                                if counter==num_of_rows:
                                    print("break")
                                    break

                if counter == num_of_rows:
                    print("break")
                    break
            print(counter)
            if counter == num_of_rows:
                self.labels = self.labels.reshape(
                    [num_of_rows, config.text_label_max_length]
                )
                print(len(self.labels))
                #break
        print("loi:"+str(len(list_of_images)))
        self.np_images = np.array(list_of_images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.np_images[idx], self.labels[idx], self.img_names[idx]

    def get_label_length_counts(self):
        return Counter(self.label_lengths)

class AHTRDataset(Dataset):
    def __init__(
        self,
        file_name,
        config,
        image_transform,
        num_of_rows,
    ):
        self.labels = torch.IntTensor()
        self.img_names = []
        self.label_lengths = []
        list_of_images = []
        counter = 0
        with open(generated_data_dir() + file_name, newline="") as file:
            reader = csv.reader(file)
            next(reader)
            text_to_int = TextToInt(config.char_to_int_map)
            fill_array = FillArray(
                length=config.text_label_max_length, empty_label=config.empty_label
            )
            for row in reader:
                try:
                    if len(row[1]) > config.text_label_max_length:
                        continue
                    if not all_chars_in_set(row[1], config.char_set):
                        continue

                    lbl_tensor = torch.IntTensor(fill_array(text_to_int(row[1])))
                    img = read_image(row[0],'RGB')
                    #print(img.shape)
                    #transform = v2.Grayscale()
                    #img = transform(img)
                    img = torchvision.transforms.functional.invert(img)
                    t = ResizeWithPad(w=156, h=44)
                    img = t(img)
                    #print(img.shape)



                    if img is None or lbl_tensor is None:
                        continue

                    list_of_images.append(img)
                    self.img_names.append(row[0])
                    self.labels = torch.cat((self.labels, lbl_tensor), 0)
                    self.label_lengths.append(len(row[1]))

                    counter = counter + 1
                    if counter == 17542:
                        print("image : ", row[0])
                        print("label : ", row[1])
                    # print("counter: ", str(counter))
                except:
                    print("error: ", str(counter))

                if counter == num_of_rows:
                    self.labels = self.labels.reshape(
                        [num_of_rows, config.text_label_max_length]
                    )
                    # self.np_images =
                    break
        self.np_images = np.array(list_of_images)
        # print("size images:", sys.getsizeof(self.images))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.np_images[idx], self.labels[idx], self.img_names[idx]

    def get_label_length_counts(self):
        return Counter(self.label_lengths)


class TransformedDatasetEpochIterator:
    def __init__(
        self,
        base_dataset,
        current_epoch,
        num_epoch,
        test_transform=A.Compose,
        train_transform=A.Compose,
        seed=0,
        train_val_split=[0.8, 0.2]
    ):
        self.base = base_dataset
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_epoch = num_epoch
        self.current_epoch = current_epoch
        random.seed(seed)
        self.random_order = random.sample(
            range(0, len(base_dataset)), len(base_dataset)
        )
        self.train_val_split = train_val_split
        self.length = len(base_dataset)

    def get_random_order(self):
        return self.random_order

    def get_epoch_ids(self):
        items_per_epoch = self.length // self.num_epoch
        num_items_gone = (self.current_epoch) * items_per_epoch

        epoch_idx_list = self.random_order[
            (self.current_epoch * items_per_epoch) : (
                self.current_epoch * items_per_epoch
            )
            + items_per_epoch
        ]
        train_size = int(len(epoch_idx_list) * self.train_val_split[0])
        # print("train size: ", str(train_size))
        val_size = len(epoch_idx_list) - train_size
        epoch_idx_train_list = epoch_idx_list[0:train_size]
        epoch_idx_val_list = epoch_idx_list[train_size:]
        return epoch_idx_train_list, epoch_idx_val_list

    def get_splits(self):
        """
        Splits the dataset into training and validation subsets.

        Returns:
            tuple: A tuple containing the training and validation subsets.
        """

        # fold_data = list(self.kf.split(self.base))
        # train_indices, val_indices = fold_data[self.current_fold]
        train_ids, val_ids = self.get_epoch_ids()
        train_data = self._get_train_subset(train_ids)
        val_data = self._get_test_subset(val_ids)

        return train_data, val_data

    def _get_train_subset(self, indices):
        return TransformedDataset(
            Subset(self.base, indices), transforms=self.train_transform
        )

    def _get_test_subset(self, indices):
        return TransformedDataset(
            Subset(self.base, indices), transforms=self.test_transform
        )




class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transforms: A.Compose):
        super(TransformedDataset, self).__init__()
        self.base = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, n = self.base[idx]
        #print(type(x))
        x_ = x.transpose(1, 2, 0)
        x_original = torchvision.transforms.functional.to_pil_image(x_)
        fname = os.path.basename(n)
        fname_no_ext = Path(fname).stem
        x_original.save("test/" + fname)
        x_original_t = self.transforms(image=x_)
        x_original_t__ = torchvision.transforms.functional.to_pil_image(x_original_t["image"])
        x_original_t__.save("test/" + fname_no_ext +"_transf"+".png")
        x_gray = v2.functional.rgb_to_grayscale(x_original_t__, num_output_channels=1)
        t=v2.ToTensor()
        x_gray=t(x_gray)
        #print(x_gray.shape)
        ft = tensor(x_gray, dtype=torch.float32)
        return ft, y

    def get_label_length_counts(self):
        print(self.base.get_label_lengths_counts())


def test_transformation(
        transformation,
        image: np.ndarray
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2)

    transformed_image = transformation(image=image)
    transformed_image = transformed_image["image"].astype(np.uint8)

    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    axes[1].imshow(transformed_image)
    axes[1].axis('off')
    axes[1].set_title('Transformed Image')

