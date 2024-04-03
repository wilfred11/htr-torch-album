import os

import torch.nn as nn
from torchvision.transforms import v2
from torchinfo import summary
from data import read_words_generate_csv, read_words_generate_bbox_csv, read_bbox_csv_show_image
from files.config import ModelConfigs
from files.dataset import ResizeWithPad, dataset_load, dataloader_show
import torch
from files.model import CRNN, BBox, visualize_model, visualize_featuremap, visualize_featuremap
from files.test_train import train
from mystuff.functions import iam_dir, ascii_dir, generated_data_dir, external_data_dir, read_maps
from wakepy import keep

# Todo confusion matrix

# th=text_transform.text_to_int('think')
# print('think', th)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_transform = v2.Compose(
    [ResizeWithPad(h=28, w=140),
     v2.Grayscale()
     ])
do = 3
text_label_max_length = 6

os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin/'

if do == 1:
    with keep.running() as k:
        print('training word reader')
        configs = read_words_generate_csv()

        char_to_int_map, int_to_char_map, char_set = read_maps()
        print('char_set', char_set)
        char_to_int_map['_'] = '15'
        int_to_char_map['15'] = '_'
        # char_to_int_map['-'] = '16'
        # int_to_char_map['16'] = '-'
        # text_label_max_length = 6

        trl, tl = dataset_load(image_transform, char_to_int_map, 1000, text_label_max_length, char_set)

        dataloader_show(trl)

        BLANK_LABEL = 15

        crnn = CRNN().to(device)
        criterion = nn.CTCLoss(blank=BLANK_LABEL, reduction='mean', zero_infinity=True)
        optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)

        # visualize_model(trl, crnn, image_transform)

        MAX_EPOCHS = 2500
        list_training_loss = []
        list_testing_loss = []

        for epoch in range(MAX_EPOCHS):
            training_loss = train(trl, crnn, optimizer, criterion, BLANK_LABEL, text_label_max_length)
            # testing_loss = test()

            list_training_loss.append(training_loss)
            # list_testing_loss.append(testing_loss)

            if epoch == 4:
                print('training loss', list_training_loss)
                # print('testing loss', list_testing_loss)
                break

        torch.save(crnn.state_dict(), generated_data_dir() + 'trained_reader')

        # https://kiran-prajapati.medium.com/hand-digit-recognition-using-recurrent-neural-network-in-pytorch-b8db24540537
        # https://medium.com/@mohini.1893/handwriting-text-recognition-236b33c5caa4
        # https://github.com/Mohini1893/Handwriting-Text-Recognition/blob/master/Initial%20Approach%202/CNN%20with%20Tensorflow%20on%20a%20character-only%20dataset.ipynb
        # https://www.youtube.com/watch?v=GxtMbmv169o
        # https://deepayan137.github.io/blog/markdown/2020/08/29/building-ocr.html
        # https://github.com/furqan4545/handwritten_text_detection_and_recognition/blob/master/handwritten_textDetectionV1.ipynb
        # https://www.youtube.com/watch?v=ZiUEdS_5Byc&t=857s
        # https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
        # https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/

        # line segmentation
        # https://towardsdatascience.com/train-a-lines-segmentation-model-using-pytorch-34d4adab8296

        # https://blog.paperspace.com/object-localization-pytorch-2/
        # https://towardsdatascience.com/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0

if do == 2:
    print('visualize featuremap')
    char_to_int_map, int_to_char_map, char_set = read_maps()
    crnn = CRNN().to(device)
    crnn.load_state_dict(torch.load(generated_data_dir() + 'trained_reader'))
    trl, tl = dataset_load(image_transform, char_to_int_map, 5, text_label_max_length, char_set)
    visualize_featuremap(crnn, trl)

if do == 3:
    char_to_int_map, int_to_char_map, char_set = read_maps()
    crnn = CRNN().to(device)
    crnn.load_state_dict(torch.load(generated_data_dir() + 'trained_reader'))
    trl, tl = dataset_load(image_transform, char_to_int_map, 5, text_label_max_length, char_set)
    visualize_model(trl, crnn)

if do == 4:
    read_words_generate_bbox_csv()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    crnn = BBox().to(device)

    summary(crnn, input_size=[1, 1, 140, 28])

    conv_weights = []  # List to store convolutional layer weights
    conv_layers = []  # List to store convolutional layers
    total_conv_layers = 0  # Counter for total convolutional layers

    for module in crnn.features.children():
        if isinstance(module, nn.Conv2d):
            total_conv_layers += 1
            conv_weights.append(module.weight)
            conv_layers.append(module)

    print(f"Total convolution layers: {total_conv_layers}")

if do == 5:
    image_transform = v2.Compose(
        [
            v2.Grayscale()
        ])
    read_bbox_csv_show_image()
