import os
import csv
import torch.nn as nn
from torchvision.transforms import v2
from torchinfo import summary
from data import read_words_generate_csv, read_words_generate_bbox_csv, read_bbox_csv_show_image
from files.config import ModelConfigs
from files.dataset import ResizeWithPad, dataset_load, dataloader_show
import torch
from files.model import CRNN, BBox, visualize_model, visualize_featuremap
from files.test_train import train
from mystuff.functions import iam_dir, ascii_dir, generated_data_dir, external_data_dir

print('start')


def read_maps():
    char_to_int_map = {}
    int_to_char_map = {}
    char_set = set()
    with open(external_data_dir() + 'char_map_15.csv', 'r') as file:
        csv_reader = csv.reader(file, delimiter=';', quotechar='\'', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for row in csv_reader:
            char_to_int_map[row[0]] = row[1]
            int_to_char_map[row[1]] = row[0]
            char_set.add(row[0])

    return char_to_int_map, int_to_char_map, char_set


# th=text_transform.text_to_int('think')
# print('think', th)
do = 1

if do == 1:

    configs = read_words_generate_csv()

    char_to_int_map, int_to_char_map, char_set = read_maps()
    char_to_int_map['_'] = '15'
    int_to_char_map['15'] = '_'
    char_to_int_map['-'] = '16'
    int_to_char_map['16'] = '-'
    '''
    char_to_int_map['\''] = '68'
    int_to_char_map['68'] = '\''
    char_to_int_map[''] = '69'
    int_to_char_map['69'] = ''
    char_to_int_map[':'] = '70'
    int_to_char_map['70'] = ':'
    char_to_int_map['('] = '71'
    int_to_char_map['71'] = '('
    char_to_int_map[')'] = '72'
    int_to_char_map['72'] = ')'
    char_to_int_map['#'] = '73'
    int_to_char_map['73'] = '#'
    char_to_int_map['?'] = '74'
    int_to_char_map['74'] = '?'
    char_to_int_map['*'] = '75'
    int_to_char_map['75'] = '*'
    char_to_int_map['/'] = '76'
    int_to_char_map['76'] = '/'
    char_to_int_map['&'] = '77'
    int_to_char_map['77'] = '&'
    char_to_int_map['+'] = '78'
    int_to_char_map['78'] = '+'
    char_to_int_map['_'] = '79'
    int_to_char_map['79'] = '_'
    
    print(char_to_int_map)
    print(int_to_char_map)
    '''

    image_transform = v2.Compose(
        [ResizeWithPad(h=28, w=140),
         v2.Grayscale()
         ])
    # v2.ToImage()])

    """label_transform = v2.Compose(
        [TextToInt(char_to_int_map),
         FillArray(configs.max_text_length)]
    )"""

    """label_transform = v2.Compose(
        [TextToInt(char_to_int_map),
          FillArray(configs.max_text_length),
         v2.ToTensor()
         ]
    )"""
    # label_transform=None

    # ds = CustomImageDataset(map=char_to_int_map ,transform=image_transform, target_transform=label_transform)
    # ds = CustomImageDataset(transform=image_transform)

    # train_set, test_set = torch.utils.data.random_split(ds, [int(len(ds) * 0.8) + 1, int(len(ds) * 0.2)])

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    text_label_max_length = 6
    trl, tl = dataset_load(image_transform, char_to_int_map, 400, text_label_max_length, char_set)



    #dataloader_show(trl)

    BLANK_LABEL = 15

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    crnn = CRNN().to(device)
    criterion = nn.CTCLoss(blank=BLANK_LABEL, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)

    #visualize_model(trl, crnn, image_transform)
    visualize_featuremap(crnn, trl)

    # train(trl, crnn, optimizer, criterion)

    MAX_EPOCHS = 250
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

    '''
    print(ds[3][1])
    print(ds[3][0])
    plt.imshow(ds[3][1])
    plt.imshow(ds[3][0].permute(1, 2, 0))
    
    plt.show()
    '''

    '''
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {len(train_features)}")
    print(f"Labels batch shape: {len(train_labels)}")
    img = train_features[1].squeeze()
    label = train_labels[1]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    '''

    # print(train_set[0][1])
    # number_of_printed_imgs = 10

    '''
    for epoch in range(1):
        for step, (x, y) in enumerate(trl):   # gives batch data
            for j in range(len(x)):
                print(y[j])
                plt.imshow(x[j], cmap='gray')
                plt.show()
    '''

    '''
    for batch_id, (x_test, y_test) in enumerate(trl):
        for j in range(len(x_test)):
            plt.imshow(x_test[j].permute(1, 2, 0), cmap='gray')
            plt.show()
    
            print(y_test[j])
            number_of_printed_imgs -= 1
    
            if number_of_printed_imgs <= 0:
                break
    
        if number_of_printed_imgs <= 0:
            break
    '''

    # print (ds[0][0])

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

if do == 3:
    image_transform = v2.Compose(
        [
         v2.Grayscale()
         ])
    read_bbox_csv_show_image()

