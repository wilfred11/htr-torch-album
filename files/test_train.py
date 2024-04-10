import torch
from itertools import groupby


def train(train_loader, crnn, optimizer, criterion, blank_label, num_chars):
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0

    for batch_id, (x_train, y_train) in enumerate(train_loader):

        batch_size = x_train.shape[0]
        crnn.reset_hidden(batch_size)

        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])

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

        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].numpy())

            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
            sz = len(prediction)
            for x in range(num_chars-sz):
                prediction = torch.cat((prediction, torch.IntTensor([16])),0)

            #print('prediction:', prediction)
            #print('y_train:', y_train[i])

            if len(prediction) == len(y_train[i]) and torch.all(prediction.eq(y_train[i])):
                correct += 1
            total += 1

        num_batches += 1


    ratio = correct / total
    print('TRAIN correct: ', correct, '/', total, ' P:', ratio)

    return total_loss / num_batches
