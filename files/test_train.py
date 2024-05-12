import jiwer
import torch
import torch.nn as nn
from itertools import groupby

from files.transform import IntToText, TextToInt


def train(train_loader, crnn, optimizer, criterion, blank_label, num_chars):
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0
    counter =0

    for batch_id, (x_train, y_train) in enumerate(train_loader):

        batch_size = x_train.shape[0]
        crnn.reset_hidden(batch_size)

        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])

        optimizer.zero_grad()

        y_pred = crnn(x_train)
        y_pred = y_pred.permute(1, 0, 2)
        #print('ypred.shp:', y_pred.shape)
        #print('y_train.shp:',y_train.shape)

        input_lengths = torch.IntTensor(batch_size).fill_(crnn.postconv_width)
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = criterion(y_pred, y_train, input_lengths, target_lengths)
        total_loss += loss.detach().numpy()

        loss.backward()
        optimizer.step()

        _, max_index = torch.max(y_pred, dim=2)
        #print('max_index.shp:', max_index.shape)
        counter=counter+1
        print('train counter:', counter)
        for i in range(batch_size):

            raw_prediction = list(max_index[:, i].numpy())
            #print('raw_prediction:', raw_prediction)
            #print('group by:', groupby(raw_prediction))
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
            #print('prediction:', prediction)
            sz = len(prediction)
            for x in range(num_chars - sz):
                prediction = torch.cat((prediction, torch.IntTensor([16])), 0)

            #print('prediction:', prediction)
            #print('y_train:', y_train[i])

            if len(prediction) == len(y_train[i]) and torch.all(prediction.eq(y_train[i])):
                correct += 1
            total += 1

        num_batches += 1

    ratio = correct / total
    print('TRAIN correct: ', correct, '/', total, ' P:', ratio)

    return total_loss / num_batches


def test(int_to_char_map, loader, crnn, optimizer, criterion, blank_label, num_chars):
    int_to_text = IntToText(int_to_char_map)
    list_of_words = list()
    list_of_hypotheses = list()
    #text_to_int = TextToInt(char_to_int_map)
    correct = 0
    total = 0
    num_batches = 0
    total_loss = 0

    for batch_id, (x_test, y_test) in enumerate(loader):
        batch_size = x_test.shape[0]
        crnn.reset_hidden(batch_size)

        x_test = x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])

        y_pred = crnn(x_test)
        y_pred = y_pred.permute(1, 0, 2)

        input_lengths = torch.IntTensor(batch_size).fill_(crnn.postconv_width)
        target_lengths = torch.IntTensor([len(t) for t in y_test])

        loss = criterion(y_pred, y_test, input_lengths, target_lengths)

        total_loss += loss.detach().numpy()

        _, max_index = torch.max(y_pred, dim=2)

        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].numpy())

            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
            prediction_as_list_of_chars = int_to_text(prediction)
            prediction_as_string = "".join([str(i) for i in prediction_as_list_of_chars])
            print('prediction:', prediction)
            print('prediction as list of chars:', prediction_as_list_of_chars)
            print('prediction as string:', prediction_as_string)
            print('y_test:', y_test[i])
            y_test_as_list_of_chars = int_to_text(y_test[i])
            print('y_test_as_list_of_chars:',y_test_as_list_of_chars)
            y_test_as_string = "".join([str(i) for i in y_test_as_list_of_chars])
            list_of_hypotheses.append(prediction_as_string)
            list_of_words.append(y_test_as_string)
            print('y_test_as_string:', y_test_as_string)
            sz = len(prediction)
            for x in range(num_chars - sz):
                prediction = torch.cat((prediction, torch.IntTensor([16])), 0)

            if len(prediction) == len(y_test[i]) and torch.all(prediction.eq(y_test[i])):
                correct += 1

            total += 1
        num_batches += 1

    ratio = correct / total
    wer = (total - correct) / total
    print('wer:', wer)
    print('TEST correct: ', correct, '/', total, ' P:', ratio)
    cer = jiwer.cer(list_of_words, list_of_hypotheses)
    print('cer:', cer)
    return total_loss / num_batches , wer, cer
