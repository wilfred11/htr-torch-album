import csv
import torch
import torchvision
import torchvision.transforms
import os
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes


from files.dataset import (
    AHTRDataset,
)
from files.replay_dataset import TransformedDatasetReplay
from files.transform import IntToText, replay_transform
from files.functions import (
    ascii_dir,
    iam_dir,
    generated_data_dir,
    htr_ds_dir,
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


def get_replay_dataset(
    config,
    num_of_rows=1000,
):
    # test_image_transform = A.Compose([])
    train_image_transform = replay_transform()
    # ))
    dataset = AHTRDataset(
        "file_names-labels.csv",
        config,
        None,
        num_of_rows,
    )
    train_set = TransformedDatasetReplay(dataset, transforms=train_image_transform)
    return train_set


def all_chars_in_set(str, set):
    for ch in str:
        if ch not in set:
            return False
    return True


def dataloader_show(loader, number_of_images, int_to_char_map):
    print("lengte dl:" + str(len(loader.dataset)))
    # print(len(loader[0]))
    for batch_id, (x_test, y_test) in enumerate(loader):
        print(str(x_test[0].shape))
        for j in range(len(x_test)):
            image = x_test[j].permute(1, 2, 0).numpy()
            # print("show: " + str(x_test[j].shape))
            plt.imshow(image, cmap="gray")
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
    with open("char_map_a_z.csv", "r") as file:
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





