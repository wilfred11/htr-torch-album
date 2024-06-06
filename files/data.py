import csv
import os
import torch
import torchvision
import torchvision.transforms
from matplotlib import pyplot as plt
from torch.utils import data as data_utils
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from files.dataset import HTRDataset
from files.transform import TextToInt, FillArray, IntToText
from files.functions import (
    ascii_dir,
    iam_dir,
    generated_data_dir,
    htr_ds_dir,
    external_data_dir,
)


def read_words_generate_csv():
    dataset = []
    words = open(os.path.join(ascii_dir(), "words.txt"), "r").readlines()
    for line in words:
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        if line_split[1] == "err":
            continue

        almost_file_name = line_split[0]
        file_name_parts = almost_file_name.split("-", 2)[:2]
        file_name = (
            file_name_parts[0]
            + "/"
            + file_name_parts[0]
            + "-"
            + file_name_parts[1]
            + "/"
            + line_split[0]
            + ".png"
        )
        label = line_split[-1].rstrip("\n")
        full_file_name = os.path.join(iam_dir(), "words", file_name)
        if not os.path.exists(full_file_name):
            print(f"File not found: {full_file_name}")
            continue

        dataset.append([full_file_name, label])

    with open(generated_data_dir() + "file_names-labels.csv", "w") as file:
        writer = csv.writer(
            file, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        writer.writerow(["file_name", "label"])
        for item in dataset:
            writer.writerow([item[0], item[1]])
            file.flush()
    file.close()
    print("file created")


def get_dataloaders(
    image_transform,
    char_to_int_map,
    int_to_char_map,
    num_of_rows,
    text_label_max_length,
    char_set,
):
    print("loading dataset")
    seq_dataset = HTRDataset(
        "file_names-labels.csv",
        text_label_max_length,
        char_to_int_map,
        int_to_char_map,
        char_set,
        image_transform,
        num_of_rows,
    )
    train_set, test_set = torch.utils.data.random_split(
        seq_dataset, [int(len(seq_dataset) * 0.8), int(len(seq_dataset) * 0.2)]
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    return train_loader, test_loader


def all_chars_in_set(str, set):
    for ch in str:
        if ch not in set:
            return False
    return True


def dataloader_show(loader, number_of_images, int_to_char_map):
    for batch_id, (x_test, y_test) in enumerate(loader):
        for j in range(len(x_test)):
            plt.imshow(x_test[j], cmap="gray")
            plt.show()

            print("word as IntTensor")
            print(y_test[j])
            transform = IntToText(int_to_char_map)
            print("word")
            print("".join(transform(y_test[j])))
            number_of_images -= 1

            if number_of_images <= 0:
                break

        if number_of_images <= 0:
            break


def read_maps():
    char_to_int_map = {}
    int_to_char_map = {}
    char_set = set()
    with open("char_map_15.csv", "r") as file:
        csv_reader = csv.reader(
            file,
            delimiter=";",
            quotechar="'",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        for row in csv_reader:
            char_to_int_map[row[0]] = row[1]
            int_to_char_map[row[1]] = row[0]
            char_set.add(row[0])

    return char_to_int_map, int_to_char_map, char_set


def read_bbox_csv_show_image():
    with open(generated_data_dir() + "form_file_names-bboxes.csv", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        current_image = ""
        last_image = ""
        t_bbox = torch.IntTensor()
        for row in reader:
            current_image = row[0]
            print("size t_bbox", len(t_bbox))
            print("cur_im", current_image)
            print("last_im", last_image)
            if (not current_image == last_image or last_image == "") and not len(
                t_bbox
            ) == 0:
                print("image shown")
                image = read_image(iam_dir() + last_image)
                current_image = ""
                last_image = row[0]
                img = draw_bounding_boxes(image, t_bbox, width=10, colors=(255, 0, 0))
                img = torchvision.transforms.ToPILImage()(img)
                img.show()
                os.system("pause")
                t_bbox = torch.IntTensor()

            print(row)
            bbox = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
            bbox = torch.tensor(bbox, dtype=torch.int)
            bbox = bbox.unsqueeze(0)
            bbox = torchvision.ops.box_convert(bbox, in_fmt="xywh", out_fmt="xyxy")

            t_bbox = torch.cat((t_bbox, bbox), 0)
            bbox = torch.Tensor()

            last_image = row[0]


def read_bbox_csv_show_image():
    with open(htr_ds_dir() + "train/" + "_annotations.csv", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        last_image = ""
        t_bbox = torch.IntTensor()
        for row in reader:
            current_image = row[0]
            if (not current_image == last_image or last_image == "") and not len(
                t_bbox
            ) == 0:
                image = read_image(htr_ds_dir() + "train/" + last_image)
                img = draw_bounding_boxes(image, t_bbox, width=5, colors=(255, 0, 0))
                img = torchvision.transforms.ToPILImage()(img)
                img.show()
                os.system("pause")
                t_bbox = torch.IntTensor()

            bbox = [int(row[4]), int(row[5]), int(row[6]), int(row[7])]
            bbox = torch.tensor(bbox, dtype=torch.int)
            bbox = bbox.unsqueeze(0)
            t_bbox = torch.cat((t_bbox, bbox), 0)
            last_image = row[0]
