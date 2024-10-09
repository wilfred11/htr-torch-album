import csv
import os
import shutil
from collections import Counter

import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch import nn
import torch.utils.data as data_utils
import albumentations as A
import torchinfo

from files.data import (
    read_words_generate_csv,
    read_bbox_csv_show_image,
    get_dataloaders,
    dataloader_show,
    read_maps,
    Aget_dataloaders,
    get_replay_dataset,
)
from files.dataset import (
    CustomObjectDetectionDataset,
    AHTRDataset,
    KFoldTransformedDatasetIterator,
)
from files.transform import ResizeWithPad, AResizeWithPad
import torch
from files.model import (
    CRNN,
    visualize_model,
    visualize_featuremap,
    CRNN_lstm,
    CRNN_rnn,
    simple_model,
    CRNN_adv,
    advanced_model,
    simple_CNN,
    advanced_CNN,
    Attention,
)
from files.test_train import train, test
from files.functions import (
    generated_data_dir,
    htr_ds_dir,
    base_no_aug_score_dir,
    base_aug_score_dir,
    aug_graphs,
    no_aug_graphs,
    adv_no_aug_score_dir,
    adv_aug_score_dir,
)
from wakepy import keep

# Todo confusion matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
image_transform = v2.Compose([ResizeWithPad(h=32, w=110), v2.Grayscale()])

do = 111
# aug = 0
# aug = 1

text_label_max_length = 8
model = 2
torch.manual_seed(1)

# models = ["gru"]
models = ["rnn"]

if do == 110:
    print("saving images and transforms")

    # train_image_transform = A.Compose([])

    read_words_generate_csv()

    char_to_int_map, int_to_char_map, char_set = read_maps()
    print("char_set", char_set)
    print("int_to_char_map", int_to_char_map)
    # char_to_int_map['_'] = '15'
    # int_to_char_map['15'] = '_'
    int_to_char_map["18"] = ""

    ds = get_replay_dataset(
        text_label_max_length,
        char_to_int_map,
        int_to_char_map,
        char_set,
    )
    ds.save_pictures_and_transform()
    ds.get_label_length_counts()

    # dataloader_show(trl, number_of_images=10, int_to_char_map=int_to_char_map)
    # dataloader_show(tl, number_of_images=10, int_to_char_map=int_to_char_map)


if do == 1:
    tfs = [
        "scores",
        "scores/adv",
        "scores/base",
        "scores/base/aug",
        "scores/adv/aug",
        "scores/base/no_aug",
        "scores/adv/no_aug",
    ]
    for tf in tfs:
        if os.path.isdir(tf):
            shutil.rmtree(tf)
        os.mkdir(tf)
    # augs = [0, 1]
    augs = [1]
    advs = [0]

    read_words_generate_csv()

    char_to_int_map, int_to_char_map, char_set = read_maps()
    int_to_char_map["18"] = ""
    print("char_set", char_set)
    print("int to char map", int_to_char_map)
    print("char to int map", char_to_int_map)

    BLANK_LABEL = 17

    for model in models:
        for adv in advs:
            for aug in augs:
                print("context: " + model + " adv: " + str(adv) + " aug: " + str(aug))
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                test_image_transform = A.Compose([])
                if aug == 1:
                    train_image_transform = A.Compose(
                        [
                            A.Rotate(limit=(-45.75, 45.75), p=1),
                            A.OneOf(
                                [
                                    A.GaussNoise(p=1),
                                    A.Blur(p=1),
                                    A.RandomGamma(p=1),
                                    A.GridDistortion(p=1),
                                    # A.PixelDropout(p=1, drop_value=None),
                                    A.Morphological(
                                        p=1, scale=(4, 6), operation="dilation"
                                    ),
                                    A.Morphological(
                                        p=1, scale=(4, 6), operation="erosion"
                                    ),
                                    A.RandomBrightnessContrast(p=1),
                                    A.Affine(p=1),
                                ],
                                p=0.50,
                            ),
                            # A.InvertImg(p=1),
                            # AResizeWithPad(h=44, w=156),
                        ]
                    )

                if aug == 0:
                    train_image_transform = A.Compose([])

                with keep.running() as k:
                    print("htr training and testing")

                    if model == "gru" and adv == 0:
                        crnn = CRNN().to(device)
                    elif model == "lstm" and adv == 0:
                        crnn = CRNN_lstm().to(device)
                    elif model == "rnn" and adv == 0:
                        crnn = CRNN_rnn().to(device)
                    elif model == "gru" and adv == 1:
                        crnn = CRNN_adv().to(device)

                    prefix = model + "_"

                    criterion = nn.CTCLoss(
                        blank=BLANK_LABEL, reduction="mean", zero_infinity=True
                    )
                    # optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)
                    optimizer = torch.optim.Adam(
                        params=crnn.parameters(),
                        lr=0.001,
                        betas=(0.9, 0.999),
                        eps=1e-08,
                        weight_decay=0,
                        amsgrad=False,
                    )

                    MAX_EPOCHS = 2500

                    dataset = AHTRDataset(
                        "file_names-labels.csv",
                        text_label_max_length,
                        char_to_int_map,
                        int_to_char_map,
                        char_set,
                        None,
                        1000,
                    )
                    # dataloader_show(trl, number_of_images=2, int_to_char_map=int_to_char_map)

                    list_training_loss = []
                    list_testing_loss = []
                    list_testing_wer = []
                    list_testing_cer = []
                    list_length_correct = []
                    trained_on_words = []

                    for fold in range(5):

                        data_handler = KFoldTransformedDatasetIterator(
                            dataset,
                            current_fold=fold,
                            num_fold=5,
                            train_transform=train_image_transform,
                            test_transform=test_image_transform,
                        )

                        train_data, test_data = data_handler.get_splits()

                        trl = torch.utils.data.DataLoader(
                            train_data, batch_size=4, shuffle=False
                        )
                        tl = torch.utils.data.DataLoader(
                            test_data, batch_size=1, shuffle=False
                        )
                        training_loss, trained_on_words = train(
                            trained_on_words,
                            int_to_char_map,
                            trl,
                            crnn,
                            optimizer,
                            criterion,
                            BLANK_LABEL,
                            text_label_max_length,
                        )
                        (
                            testing_loss,
                            wer,
                            cer,
                            length_correct,
                            list_words,
                            list_hypotheses,
                        ) = test(
                            int_to_char_map,
                            tl,
                            crnn,
                            optimizer,
                            criterion,
                            BLANK_LABEL,
                            text_label_max_length,
                        )

                        list_training_loss.append(training_loss)
                        list_testing_loss.append(testing_loss)
                        list_testing_wer.append(wer)
                        list_testing_cer.append(cer)
                        list_length_correct.append(length_correct)

                        if aug == 0 and adv == 0:
                            dir = base_no_aug_score_dir()
                        elif aug == 1 and adv == 0:
                            dir = base_aug_score_dir()
                        elif aug == 0 and adv == 1:
                            dir = adv_no_aug_score_dir()
                        elif aug == 1 and adv == 1:
                            dir = adv_aug_score_dir()

                        columns = ["word", "hypothesis"]
                        with open(
                            dir
                            + prefix
                            + "words_hypothesis_fold_"
                            + str(fold)
                            + ".csv",
                            "w",
                            newline="",
                        ) as f:
                            write = csv.writer(f)
                            write.writerow(columns)
                            for i in range(len(list_words)):
                                l = [list_words[i], list_hypotheses[i]]
                                write.writerow(l)

                        if fold == 4:
                            print("training loss", list_training_loss)
                            with open(
                                dir + prefix + "list_training_loss.pkl", "wb"
                            ) as f1:
                                pickle.dump(list_training_loss, f1)
                            print("testing loss", list_testing_loss)
                            with open(
                                dir + prefix + "list_testing_loss.pkl", "wb"
                            ) as f2:
                                pickle.dump(list_testing_loss, f2)
                            with open(
                                dir + prefix + "list_testing_wer.pkl", "wb"
                            ) as f3:
                                pickle.dump(list_testing_wer, f3)
                            with open(
                                dir + prefix + "list_testing_cer.pkl", "wb"
                            ) as f4:
                                pickle.dump(list_testing_cer, f4)
                            with open(
                                dir + prefix + "list_testing_length_correct.pkl",
                                "wb",
                            ) as f5:
                                pickle.dump(list_length_correct, f5)

                            trained_on_words_count = dict(Counter(trained_on_words))
                            with open(
                                dir + prefix + "trained_on_words_count.csv",
                                "w",
                                newline="",
                            ) as f6:
                                w = csv.writer(f6)
                                w.writerows(trained_on_words_count.items())

                            break

                    torch.save(crnn.state_dict(), dir + prefix + "trained_reader")
if do == 111:
    cnn = simple_CNN()
    torchinfo.summary(
        cnn,
        input_size=(1, 1, 156, 44),
    )
    adv_cnn = advanced_CNN()
    torchinfo.summary(
        adv_cnn,
        input_size=(1, 1, 156, 44),
    )

    attention = Attention(128)
    torchinfo.summary(
        attention,
        input_size=(35, 128),
    )


if do == 2:
    print("visualize featuremap")
    char_to_int_map, int_to_char_map, char_set = read_maps()
    crnn = CRNN().to(device)
    crnn.load_state_dict(torch.load(generated_data_dir() + "trained_reader"))
    trl, _ = get_dataloaders(
        image_transform,
        char_to_int_map,
        int_to_char_map,
        5,
        text_label_max_length,
        char_set,
    )
    visualize_featuremap(crnn, trl, 1)

if do == 3:
    print("visualize model")
    char_to_int_map, int_to_char_map, char_set = read_maps()
    crnn = CRNN().to(device)
    crnn.load_state_dict(torch.load(generated_data_dir() + "trained_reader"))
    trl, tl = get_dataloaders(
        image_transform,
        char_to_int_map,
        int_to_char_map,
        5,
        text_label_max_length,
        char_set,
    )
    visualize_model(trl, crnn)


if do == 6:

    if model == 2:
        prefix = "gru_"
    elif model == 3:
        prefix = "lstm_"
    elif model == 1:
        prefix = "rnn_"

    if aug == 0:
        dir = base_no_aug_score_dir()
    else:
        dir = base_aug_score_dir()

    with open(dir + prefix + "list_training_loss.pkl", "rb") as f1:
        list_training_loss = pickle.load(f1)
    with open(dir + prefix + "list_testing_loss.pkl", "rb") as f2:
        list_testing_loss = pickle.load(f2)

    epochs = range(1, len(list_training_loss) + 1)
    plt.plot(epochs, list_training_loss, "g", label="Training loss")
    plt.plot(epochs, list_testing_loss, "b", label="Testing loss")
    plt.xticks(range(1, len(list_training_loss) + 1))
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if do == 61:
    tfs = ["aug", "no_aug", "aug/graphs", "no_aug/graphs"]
    for tf in tfs:
        if os.path.isdir(tf):
            shutil.rmtree(tf)
        os.mkdir(tf)

    prefix = ""
    if model == 2:
        prefix = "gru"
    elif model == 3:
        prefix = "lstm"
    elif model == 1:
        prefix = "rnn"

    if aug == 0:
        dir = base_no_aug_score_dir()
    else:
        dir = base_aug_score_dir()

    with open(dir + prefix + "list_testing_wer.pkl", "rb") as f3:
        gru_list_testing_wer = pickle.load(f3)
    with open(dir + prefix + "list_testing_cer.pkl", "rb") as f4:
        gru_list_testing_cer = pickle.load(f4)
    with open(dir + "lstm_list_testing_wer.pkl", "rb") as f5:
        lstm_list_testing_wer = pickle.load(f5)
    with open(dir + "lstm_list_testing_cer.pkl", "rb") as f6:
        lstm_list_testing_cer = pickle.load(f6)
    with open(dir + "rnn_list_testing_wer.pkl", "rb") as f1:
        rnn_list_testing_wer = pickle.load(f1)
    with open(dir + "rnn_list_testing_cer.pkl", "rb") as f2:
        rnn_list_testing_cer = pickle.load(f2)
    epochs = range(1, len(lstm_list_testing_wer) + 1)
    plt.plot(epochs, gru_list_testing_wer, label="CRNN GRU testing wer", color="black")
    plt.plot(epochs, gru_list_testing_cer, label="CRNN GRU testing cer", color="blue")
    plt.plot(epochs, lstm_list_testing_wer, label="CRNN LSTM testing wer", color="red")
    plt.plot(
        epochs, lstm_list_testing_cer, label="CRNN LSTM testing cer", color="orange"
    )
    plt.plot(epochs, rnn_list_testing_wer, label="CRNN RNN testing wer", color="grey")
    plt.plot(epochs, rnn_list_testing_cer, label="CRNN RNN testing cer", color="green")
    plt.xticks(range(1, len(lstm_list_testing_wer) + 1))
    # plt.title('Testing wer/cer')
    plt.xlabel("Epochs")
    plt.ylabel("wer/cer")
    plt.legend()
    plt.show()
    if aug == 0:
        plt.savefig(no_aug_graphs() + "compare models.png")
    else:
        plt.savefig(aug_graphs() + "compare models.png")
