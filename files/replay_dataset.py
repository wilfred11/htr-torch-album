import os
import random
import shutil

import albumentations as A
import torch
import torchvision
from torch import tensor
from torch.utils.data import Dataset


class TransformedDatasetReplay(Dataset):
    def __init__(self, base_dataset, transforms: A.ReplayCompose):
        super(TransformedDatasetReplay, self).__init__()
        self.base = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, img_name = self.base[idx]
        f_t = self.transforms(image=x)
        ft = tensor(f_t["image"], dtype=torch.float32)
        return ft, y


    def save_pictures_and_transform1(self):
        target_folder = "images_and_transform/"
        if os.path.isdir(target_folder):
            shutil.rmtree(target_folder)
        os.mkdir(target_folder)

        rnd_indexes = [random.randint(0, len(self.base)) for p in range(0, 10)]
        count = 0
        # print(x)
        f = open(target_folder + "transforms_.txt", "a")
        for i in rnd_indexes:
            print(type(self.base[i][0]))
            print(self.base[i][0].shape)
            x_bt = self.base[i][0]
            #x_bt = x_bt.permute(0, 1, 2)
            x_bt=x_bt.transpose(1, 2, 0)
            x_original = torchvision.transforms.functional.to_pil_image(x_bt)
            fname = os.path.basename(self.base[i][2])
            x_original.save(target_folder + fname)
            x_t = self.transforms(image=x_bt)
            x_transformed = torchvision.transforms.functional.to_pil_image(x_t['image'], mode='RGB')
            x_transformed.save(target_folder +'transf_'+ fname)
            print(x_t["replay"]["transforms"][0])
            f.write(fname+"\n")
            f.write(x_t["replay"]["transforms"][0]["__class_fullname__"] + "\n")
            f.write(str(x_t["replay"]["transforms"][0]["params"]["matrix"][0][2]) + "\n")

            for one_of in x_t["replay"]["transforms"][1]["transforms"]:
                if one_of["applied"]:
                    # print(one_of["__class_fullname__"])
                    f.write(str(one_of["__class_fullname__"]) + "\n")
                    if one_of["__class_fullname__"] == "Morphological":
                        f.write(str(one_of["operation"]) + "\n")
                    # f.write(one_of)
                    # print(one_of)"""
            """for one_of in x_t["replay"]["transforms"][2]["transforms"]:
                if one_of["applied"]:
                    # print(one_of["__class_fullname__"])
                    f.write(str(one_of["__class_fullname__"]) + "\n")"""
            f.write("xxxxxxxxxxx\n")
        f.close()

    def get_label_length_counts(self):
        print(self.base.get_label_length_counts())
