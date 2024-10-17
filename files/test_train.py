import jiwer
import torch
from itertools import groupby

from files.transform import IntToText, TextToInt, IntToString


def train(trained_on_words, train_loader, crnn, optimizer, criterion, config):
    int_to_string = IntToString(config.int_to_char_map)
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0
    counter = 0

    for batch_id, (x_train, y_train) in enumerate(train_loader):

        batch_size = x_train.shape[0]
        crnn.reset_hidden(batch_size)
        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[2], x_train.shape[3])

        optimizer.zero_grad()

        y_pred = crnn(x_train)
        y_pred = y_pred.permute(1, 0, 2)

        input_lengths = torch.IntTensor(batch_size).fill_(crnn.postconv_width)
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = criterion(y_pred, y_train, input_lengths, target_lengths)
        total_loss += loss.detach().numpy()

        loss.backward()
        optimizer.step()

        _, max_index = torch.max(y_pred, dim=2)
        counter = counter + 1
        # print('train counter:', counter)
        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].numpy())
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != config.blank_label]
            )
            prediction_as_string = int_to_string(prediction)

            # print('prediction:', prediction)
            sz = len(prediction)
            for x in range(config.text_label_max_length - sz):
                prediction = torch.cat(
                    (prediction, torch.IntTensor([config.empty_label])), 0
                )
                # print(prediction)

            if len(prediction) == len(y_train[i]) and torch.all(
                prediction.eq(y_train[i])
            ):
                correct += 1

            y_train__as_string = int_to_string(y_train[i])
            # print(y_train__as_string)
            trained_on_words.append(y_train__as_string)

            total += 1

        num_batches += 1

    ratio = correct / total
    print("TRAIN correct: ", correct, "/", total, " P:", ratio)

    return total_loss / num_batches, trained_on_words


def int_tensor_to_string(int_to_char_map, int_tensor):
    int_to_text = IntToText(int_to_char_map)
    list_of_chars = int_to_text(int_tensor)
    string = "".join([str(c) for c in list_of_chars])
    return string


def test(loader, crnn, optimizer, criterion, config):
    int_to_string = IntToString(config.int_to_char_map)
    list_of_words = list()
    list_of_hypotheses = list()
    list_of_lengths_and_correctness = {}
    correct = 0
    total = 0
    num_batches = 0
    total_loss = 0

    for batch_id, (x_test, y_test) in enumerate(loader):
        batch_size = x_test.shape[0]
        crnn.reset_hidden(batch_size)

        x_test = x_test.view(x_test.shape[0], 1, x_test.shape[2], x_test.shape[3])

        y_pred = crnn(x_test)
        # print(y_pred)
        y_pred = y_pred.permute(1, 0, 2)
        # print(y_pred)

        input_lengths = torch.IntTensor(batch_size).fill_(crnn.postconv_width)
        target_lengths = torch.IntTensor([len(t) for t in y_test])

        loss = criterion(y_pred, y_test, input_lengths, target_lengths)

        total_loss += loss.detach().numpy()

        _, max_index = torch.max(y_pred, dim=2)

        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].numpy())
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != config.blank_label]
            )
            prediction_as_string = int_to_string(prediction)
            y_test_as_string = int_to_string(y_test[i])
            list_of_hypotheses.append(prediction_as_string)
            list_of_words.append(y_test_as_string)

            # print(prediction_as_string)
            # print(y_test_as_string)
            if prediction_as_string == y_test_as_string:
                list_of_lengths_and_correctness[(len(y_test_as_string), "correct")] = (
                    list_of_lengths_and_correctness[(len(y_test_as_string), "correct")]
                    + 1
                    if list_of_lengths_and_correctness.get(
                        (len(y_test_as_string), "correct")
                    )
                    is not None
                    else 1
                )
            else:
                list_of_lengths_and_correctness[
                    (len(y_test_as_string), "incorrect")
                ] = (
                    list_of_lengths_and_correctness[
                        (len(y_test_as_string), "incorrect")
                    ]
                    + 1
                    if list_of_lengths_and_correctness.get(
                        (len(y_test_as_string), "incorrect")
                    )
                    is not None
                    else 1
                )

            sz = len(prediction)
            for x in range(config.text_label_max_length - sz):
                prediction = torch.cat(
                    (prediction, torch.IntTensor([config.empty_label])), 0
                )

            if len(prediction) == len(y_test[i]) and torch.all(
                prediction.eq(y_test[i])
            ):
                correct += 1

            total += 1
        num_batches += 1

    ratio = correct / total
    wer = (total - correct) / total
    cer = jiwer.cer(list_of_words, list_of_hypotheses)
    print("wer:", wer)
    print("cer:", cer)
    lolc = dict(sorted(list_of_lengths_and_correctness.items()))
    # print("length and correctness: ", lolc)
    print("TEST correct: ", correct, "/", total, " P:", ratio)

    return total_loss / num_batches, wer, cer, lolc, list_of_words, list_of_hypotheses
