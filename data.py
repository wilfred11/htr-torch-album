import csv
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from matplotlib import patches
from matplotlib.patches import Rectangle
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from files.config import ModelConfigs
from mystuff.functions import ascii_dir, iam_dir, generated_data_dir, htr_ds_dir


def read_words_generate_csv1():
    dataset, vocab, max_len = [], set(), 0
    # Preprocess the dataset by the specific IAM_Words dataset file structure
    words = open(os.path.join(ascii_dir(), "words.txt"), "r").readlines()
    for line in words:
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        if line_split[1] == "err":
            continue

        almost_file_name = line_split[0]

        folder1 = line_split[0][:3]
        print('folder1:', folder1)
        folder2 = "-".join(line_split[0].split("-")[:2])
        file_name = line_split[0] + ".png"
        label = line_split[-1].rstrip('\n')

        rel_path = os.path.join(iam_dir(), "words", folder1, folder2, file_name)
        if not os.path.exists(rel_path):
            print(f"File not found: {rel_path}")
            continue

        dataset.append([rel_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

    configs = ModelConfigs()

    # Save vocab and maximum text length to configs
    configs.vocab = "".join(sorted(vocab))
    configs.max_text_length = max_len
    configs.save()

    with open(generated_data_dir() + 'file_names-labels.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        # writer.writerow(dataset)
        writer.writerow(['file_name', 'label'])
        for item in dataset:
            # Write item to outcsv
            writer.writerow([item[0], item[1]])

    return configs


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
        file_name = file_name_parts[0] + '/' + file_name_parts[0] + '-' + file_name_parts[1] + '/' + line_split[0] + '.png'
        label = line_split[-1].rstrip('\n')
        full_file_name = os.path.join(iam_dir(), "words", file_name)
        if not os.path.exists(full_file_name):
            print(f"File not found: {full_file_name}")
            continue

        dataset.append([file_name, label])

    with open(generated_data_dir() + 'file_names-labels.csv', 'w') as file:
         writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
         writer.writerow(['file_name', 'label'])
         for item in dataset:
             writer.writerow([item[0], item[1]])
             file.flush()
    file.close()


def read_words_generate_bbox_csv():
    dataset = []
    # Preprocess the dataset by the specific IAM_Words dataset file structure
    words = open(os.path.join(ascii_dir(), "words.txt"), "r").readlines()
    for line in words:
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        # if line_split[1] == "err":
        #    continue

        almost_file_name = line_split[0]
        r = almost_file_name.split("-", 2)[:2]
        file_name = r[0] + '-' + r[1]
        if file_name[0] in ['a', 'b', 'c', 'd']:
            file_name = 'formsA-D' + '/' + file_name
        elif file_name[0] in ['e', 'f', 'g', 'h']:
            file_name = 'formsE-H' + '/' + file_name
        elif file_name[0] in ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
            file_name = 'formsI-Z' + '/' + file_name

        print('filename:', file_name)
        file_name = file_name + ".png"

        bbox = line_split[3:7]
        print('bbox:', bbox)

        dataset.append([file_name, bbox[0], bbox[1], bbox[2], bbox[3]])

    with open(generated_data_dir() + 'form_file_names-bboxes.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['file_name', 'x', 'y', 'w', 'h'])
        for item in dataset:
            writer.writerow([item[0], item[1], item[2], item[3], item[4]])


def read_lines_generate_bbox_csv():
    dataset = []
    # Preprocess the dataset by the specific IAM_Words dataset file structure
    words = open(os.path.join(ascii_dir(), "lines.txt"), "r").readlines()
    for line in words:
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        # if line_split[1] == "err":
        #    continue

        almost_file_name = line_split[0]
        r = almost_file_name.split("-", 2)[:2]
        file_name = r[0] + '/' + r[0] + '-' + r[1] + '/' + line_split[0]
        '''
        if file_name[0] in ['a', 'b', 'c', 'd']:
            file_name = 'formsA-D' + '/' + file_name
        elif file_name[0] in ['e', 'f', 'g', 'h']:
            file_name = 'formsE-H' + '/' + file_name
        elif file_name[0] in ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
            file_name = 'formsI-Z' + '/' + file_name
        '''

        print('filename:', file_name)
        file_name = file_name + ".png"

        bbox = line_split[3:7]
        print('bbox:', bbox)

        dataset.append([file_name, bbox[0], bbox[1], bbox[2], bbox[3]])

    with open(generated_data_dir() + 'line_file_names-bboxes.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['file_name', 'x', 'y', 'w', 'h'])
        for item in dataset:
            writer.writerow([item[0], item[1], item[2], item[3], item[4]])


def read_bbox_csv_show_image():
    with open(generated_data_dir() + 'form_file_names-bboxes.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        current_image = ''
        last_image = ''
        t_bbox = torch.IntTensor()
        for row in reader:
            current_image = row[0]
            print('size t_bbox', len(t_bbox))
            print('cur_im', current_image)
            print('last_im', last_image)
            if (not current_image == last_image or last_image == '') and not len(t_bbox) == 0:
                print('image shown')
                image = read_image(iam_dir() + last_image)
                current_image = ''
                last_image = row[0]
                img = draw_bounding_boxes(image, t_bbox, width=10, colors=(255, 0, 0))
                img = torchvision.transforms.ToPILImage()(img)
                img.show()
                os.system('pause')
                t_bbox = torch.IntTensor()

            print(row)
            bbox = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
            bbox = torch.tensor(bbox, dtype=torch.int)
            bbox = bbox.unsqueeze(0)
            bbox = torchvision.ops.box_convert(bbox, in_fmt='xywh', out_fmt='xyxy')

            t_bbox = torch.cat((t_bbox, bbox), 0)
            bbox = torch.Tensor()

            last_image = row[0]


def read_bbox_csv_show_image():
    with open(htr_ds_dir() + 'train/' + '_annotations.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        current_image = ''
        last_image = ''
        t_bbox = torch.IntTensor()
        for row in reader:
            current_image = row[0]
            print('size t_bbox', len(t_bbox))
            print('cur_im', current_image)
            print('last_im', last_image)
            if (not current_image == last_image or last_image == '') and not len(t_bbox) == 0:
                print('image shown')
                image = read_image(htr_ds_dir() + 'train/' + last_image)
                current_image = ''
                last_image = row[0]
                img = draw_bounding_boxes(image, t_bbox, width=5, colors=(255, 0, 0))
                img = torchvision.transforms.ToPILImage()(img)
                img.show()
                os.system('pause')
                t_bbox = torch.IntTensor()

            print(row)
            bbox = [int(row[4]), int(row[5]), int(row[6]), int(row[7])]
            bbox = torch.tensor(bbox, dtype=torch.int)
            bbox = bbox.unsqueeze(0)
            # bbox = torchvision.ops.box_convert(bbox, in_fmt='xywh', out_fmt='xyxy')

            t_bbox = torch.cat((t_bbox, bbox), 0)
            bbox = torch.Tensor()

            last_image = row[0]


def read_bbox_csv_to_dataset():
    with open(htr_ds_dir() + 'train/' + '_annotations.csv', newline='') as file:
        reader = csv.reader(file)
        t_images = torch.FloatTensor()
        t_bboxes = torch.IntTensor()
        t_labels = torch.IntTensor()
        next(reader)
        current_image = ''
        last_image = ''
        t_bbox = torch.IntTensor()
        for row in reader:
            current_image = row[0]
            print('size t_bbox', len(t_bbox))
            print('cur_im', current_image)
            print('last_im', last_image)
            if (not current_image == last_image or last_image == '') and not len(t_bbox) == 0:
                print('image shown')
                image = read_image(htr_ds_dir() + 'train/' + last_image)
                current_image = ''
                last_image = row[0]
                t_images= torch.cat((t_images, image),0)

                t_bboxes = torch.IntTensor()

            print(row)
            bbox = [int(row[4]), int(row[5]), int(row[6]), int(row[7])]
            bbox = torch.tensor(bbox, dtype=torch.int)
            bbox = bbox.unsqueeze(0)
            # bbox = torchvision.ops.box_convert(bbox, in_fmt='xywh', out_fmt='xyxy')

            t_bboxes = torch.cat((t_bbox, bbox), 0)
            t_labels = torch.cat(t_labels, 0)
            #bbox = torch.Tensor()

            last_image = row[0]





def htr_ds_read_bbox_csv_to_dataset():
    with open(htr_ds_dir() + 'train/' + '_annotations.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        current_image = ''
        last_image = ''
        t_bbox = torch.IntTensor()
        for row in reader:
            current_image = row[0]
            print('size t_bbox', len(t_bbox))
            print('cur_im', current_image)
            print('last_im', last_image)
            if (not current_image == last_image or last_image == '') and not len(t_bbox) == 0:
                print('image shown')
                image = read_image(htr_ds_dir() + 'train/' + last_image)
                current_image = ''
                last_image = row[0]
                img = draw_bounding_boxes(image, t_bbox, width=5, colors=(255, 0, 0))
                img = torchvision.transforms.ToPILImage()(img)
                img.show()
                os.system('pause')
                t_bbox = torch.IntTensor()

            print(row)
            bbox = [int(row[4]), int(row[5]), int(row[6]), int(row[7])]
            bbox = torch.tensor(bbox, dtype=torch.int)
            bbox = bbox.unsqueeze(0)
            # bbox = torchvision.ops.box_convert(bbox, in_fmt='xywh', out_fmt='xyxy')

            t_bbox = torch.cat((t_bbox, bbox), 0)
            bbox = torch.Tensor()

            last_image = row[0]
