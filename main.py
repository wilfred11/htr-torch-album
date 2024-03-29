import os
import csv
import torch.nn as nn
from torchvision.transforms import v2, InterpolationMode
from files.config import ModelConfigs
from files.dataset import ResizeWithPad, dataset_load, dataloader_show
import torch
from files.model import CRNN
from files.test_train import train
from mystuff.functions import iam_dir, ascii_dir, generated_data_dir, external_data_dir

print('start')


def read_maps():
    char_to_int_map = {}
    int_to_char_map = {}
    char_set = set()
    with open(external_data_dir() + 'char_map_lim.csv', 'r') as file:
        csv_reader = csv.reader(file, delimiter=';', quotechar='\'', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for row in csv_reader:
            char_to_int_map[row[0]] = row[1]
            int_to_char_map[row[1]] = row[0]
            char_set.add(row[0])

    return char_to_int_map, int_to_char_map, char_set


def read_words_generate_csv():
    dataset, vocab, max_len = [], set(), 0
    # Preprocess the dataset by the specific IAM_Words dataset file structure
    words = open(os.path.join(ascii_dir(), "words.txt"), "r").readlines()
    for line in words:
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        if line_split[1] == "err":
            continue

        folder1 = line_split[0][:3]
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


# th=text_transform.text_to_int('think')
# print('think', th)


configs = read_words_generate_csv()

char_to_int_map, int_to_char_map, char_set = read_maps()
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
    [ResizeWithPad(h=56, w=188),
     v2.Grayscale(),
    ])
     #v2.ToImage()])

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
trl, tl = dataset_load(image_transform, char_to_int_map, 4000, text_label_max_length, char_set)

dataloader_show(trl)

BLANK_LABEL = 26

device = 'cuda' if torch.cuda.is_available() else 'cpu'
crnn = CRNN().to(device)
criterion = nn.CTCLoss(blank=BLANK_LABEL, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(crnn.parameters(), lr=0.01)

#train(trl, crnn, optimizer, criterion)



MAX_EPOCHS = 100
list_training_loss = []
list_testing_loss = []

for epoch in range(MAX_EPOCHS):
    training_loss = train(trl, crnn, optimizer, criterion, BLANK_LABEL)
    #testing_loss = test()

    list_training_loss.append(training_loss)
    #list_testing_loss.append(testing_loss)

    if epoch == 4:
        print('training loss', list_training_loss)
        #print('testing loss', list_testing_loss)
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
