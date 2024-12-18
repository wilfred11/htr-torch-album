import csv
import os
import random
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

from files.config import Config
from files.data import (
    read_words_generate_csv,
    read_bbox_csv_show_image,
    get_dataloaders,
    dataloader_show,
    read_maps,
    get_replay_dataset,
)
from files.dataset import (
    CustomObjectDetectionDataset,
    AHTRDataset,
    KFoldTransformedDatasetIterator,
    TransformedDatasetEpochIterator,
)
from files.transform import ResizeWithPad, AResizeWithPad, train_transform
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

do = 1
# aug = 0
# aug = 1

text_label_max_length = 8
model = 2
torch.manual_seed(1)

random_seed = 1
random.seed(random_seed)

# models = ["gru"]
models = ["gru"]

if do == 110:
    print("saving images and transforms")
    read_words_generate_csv()

    config = Config("char_map_15.csv", 10)

    ds = get_replay_dataset(config)
    ds.save_pictures_and_transform()
    ds.get_label_length_counts()


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

    config = Config("char_map_short.csv", 10)

    print("num classes: ", config.num_classes)
    print("blank_label: ", config.blank_label)
    print("empty_label: ", config.empty_label)
    print("char_set: ", config.char_set)
    print("int to char map: ", config.int_to_char_map)
    print("char to int map: ", config.char_to_int_map)

    for model in models:
        for adv in advs:
            for aug in augs:
                print("context: " + model + " adv: " + str(adv) + " aug: " + str(aug))
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                test_image_transform = A.Compose([])
                if aug == 1:
                    train_image_transform = train_transform()

                if aug == 0:
                    train_image_transform = A.Compose([])

                with keep.running() as k:
                    print("htr training and testing")

                    if model == "gru" and adv == 0:
                        crnn = CRNN(config.num_classes).to(device)
                    elif model == "lstm" and adv == 0:
                        crnn = CRNN_lstm(config.num_classes).to(device)
                    elif model == "rnn" and adv == 0:
                        crnn = CRNN_rnn(config.num_classes).to(device)
                    elif model == "gru" and adv == 1:
                        crnn = CRNN_adv(config.num_classes).to(device)

                    prefix = model + "_"

                    criterion = nn.CTCLoss(
                        blank=config.blank_label, reduction="mean", zero_infinity=True
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

                    # MAX_EPOCHS = 2500

                    dataset = AHTRDataset(
                        "file_names-labels.csv",
                        config,
                        None,
                        5000,
                    )
                    print("length ds: ", str(len(dataset)))
                    # dataloader_show(trl, number_of_images=2, int_to_char_map=int_to_char_map)

                    list_training_loss = []
                    list_testing_loss = []
                    list_testing_wer = []
                    list_testing_cer = []
                    list_length_correct = []
                    trained_on_words = []

                    for epoch in range(config.num_epoch):
                        data_handler = TransformedDatasetEpochIterator(
                            dataset,
                            current_epoch=epoch,
                            num_epoch=config.num_epoch,
                            train_transform=train_image_transform,
                            test_transform=test_image_transform,
                            seed=random_seed,
                        )
                        train_data, test_data = data_handler.get_splits()

                        trl = torch.utils.data.DataLoader(
                            train_data, batch_size=4, shuffle=False
                        )
                        tl = torch.utils.data.DataLoader(
                            test_data, batch_size=1, shuffle=False
                        )
                        training_loss, trained_on_words = train(
                            trained_on_words, trl, crnn, optimizer, criterion, config
                        )
                        (
                            testing_loss,
                            wer,
                            cer,
                            length_correct,
                            list_words,
                            list_hypotheses,
                        ) = test(tl, crnn, optimizer, criterion, config)

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
                            + "words_hypothesis_epoch_"
                            + str(epoch)
                            + ".csv",
                            "w",
                            newline="",
                        ) as f:
                            write = csv.writer(f)
                            write.writerow(columns)
                            for i in range(len(list_words)):
                                l = [list_words[i], list_hypotheses[i]]
                                write.writerow(l)

                        if epoch == 4:
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

                            trained_on_words_count = dict(
                                sorted(
                                    trained_on_words_count.items(),
                                    key=lambda item: len(item[0]),
                                )
                            )

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
