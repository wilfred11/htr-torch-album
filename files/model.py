import os
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchlens as tl


class simple_model(nn.Module):

    def __init__(self):
        super(simple_model, self).__init__()

        self.num_classes = 18 + 1
        self.image_H = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.in1 = nn.InstanceNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.in2 = nn.InstanceNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
        self.in3 = nn.InstanceNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.in4 = nn.InstanceNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.in5 = nn.InstanceNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2)
        self.in6 = nn.InstanceNorm2d(64)
        # http://layer-calc.com/
        # c= 64 h=10 w=43

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.in1(out)

        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = self.in2(out)

        out = self.conv3(out)
        out = F.leaky_relu(out)
        out = self.in3(out)

        out = self.conv4(out)
        out = F.leaky_relu(out)
        out = self.in4(out)

        out = self.conv5(out)
        out = F.leaky_relu(out)
        out = self.in5(out)

        out = self.conv6(out)
        out = F.leaky_relu(out)
        out = self.in6(out)
        # print('out.shp before perm:', out.shape)

        # print('final out.shp:', out.shape)
        return out


class advanced_model(nn.Module):

    def __init__(self):
        super(advanced_model, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3))
        self.in1 = nn.InstanceNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.in2 = nn.InstanceNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2)
        self.in3 = nn.InstanceNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.in4 = nn.InstanceNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.in5 = nn.InstanceNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2)
        self.in6 = nn.InstanceNorm2d(128)
        # http://layer-calc.com/
        # http://layer-calc.com/
        # c= 64 h=10 w=43

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.in1(out)

        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = self.in2(out)

        out = self.conv3(out)
        out = F.leaky_relu(out)
        out = self.in3(out)

        out = self.conv4(out)
        out = F.leaky_relu(out)
        out = self.in4(out)

        out = self.conv5(out)
        out = F.leaky_relu(out)
        out = self.in5(out)

        out = self.conv6(out)
        out = F.leaky_relu(out)
        out = self.in6(out)
        # print('out.shp before perm:', out.shape)

        # print('final out.shp:', out.shape)
        return out


class AdaptiveCNN(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Here, `module` is an instance of a `Module` subclass. We save it
            # in the member variable `_modules` of the `Module` class, and its
            # type is OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            X = block(X)
        return X


def advanced_CNN():
    adv = AdaptiveCNN(
        nn.Conv2d(1, 64, kernel_size=(3, 3)),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 64, kernel_size=(3, 3)),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 128, kernel_size=(3, 3)),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(),
        nn.Conv2d(128, 128, kernel_size=(3, 3)),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(),
    )
    return adv


def simple_CNN():
    simple = AdaptiveCNN(
        nn.Conv2d(1, 32, kernel_size=(3, 3)),
        nn.InstanceNorm2d(32),
        nn.LeakyReLU(),
        nn.Conv2d(32, 32, kernel_size=(3, 3)),
        nn.InstanceNorm2d(32),
        nn.LeakyReLU(),
        nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2),
        nn.InstanceNorm2d(32),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, kernel_size=(3, 3)),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 64, kernel_size=(3, 3)),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(),
    )
    return simple


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states):
        scores = self.attention(hidden_states)
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        return context_vector


class BeamSearch(nn.Module):
    def __init__(self):
        super(BeamSearch, self).__init__()
        pass

    def forward(self, x):
        values, indexes = torch.topk(x, 3)
        print("values: ", values)
        print("indexes: ", indexes)
        return values, indexes


class CRNN_adv(nn.Module):

    def __init__(self, num_classes):
        super(CRNN_adv, self).__init__()

        self.num_classes = num_classes + 1
        self.image_H = 44

        self.cnn = advanced_CNN()
        # http://layer-calc.com/
        # c= 64 h=10 w=43

        self.postconv_height = 7
        self.postconv_width = 35

        self.gru_input_size = self.postconv_height * 64
        # self.gru_hidden_size = 128
        self.gru_hidden_size = 192
        # self.gru_num_layers = 2
        self.gru_num_layers = 4
        self.gru_h = None
        self.gru_cell = None

        self.gru = nn.GRU(
            self.gru_input_size,
            self.gru_hidden_size,
            self.gru_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(self.gru_hidden_size * 2, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.cnn(x)
        # print('out.shp before perm:', out.shape)
        out = out.permute(0, 3, 2, 1)
        # print('out.shp:', out.shape)
        # print('out:', out)
        out = out.reshape(batch_size, -1, self.gru_input_size)
        # print('out after resh:', out.shape)

        out, gru_h = self.gru(out, self.gru_h)
        # print('gru_h.shp:',gru_h.shape)
        self.gru_h = gru_h.detach()
        # print(gru_h.shape)
        # print(out.shape)
        out = torch.stack(
            [F.log_softmax(self.fc(out[i]), 1) for i in range(out.shape[0])]
        )
        # print('final out.shp:', out.shape)
        return out

    def reset_hidden(self, batch_size):
        h = torch.zeros(self.gru_num_layers * 2, batch_size, self.gru_hidden_size)
        self.gru_h = Variable(h)

    def simple_forward(self, x):
        out = self.cnn(x)
        return out.permute(0, 2, 3, 1).detach().numpy()


class CRNN(nn.Module):

    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.num_classes = num_classes + 1
        self.image_H = 44

        self.cnn = simple_CNN()
        # http://layer-calc.com/
        # c= 64 h=10 w=43

        self.postconv_height = 7
        self.postconv_width = 35

        self.gru_input_size = self.postconv_height * 64
        self.gru_hidden_size = 128
        self.gru_num_layers = 2
        self.gru_h = None
        self.gru_cell = None

        self.gru = nn.GRU(
            self.gru_input_size,
            self.gru_hidden_size,
            self.gru_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(self.gru_hidden_size * 2, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.cnn(x)
        # print('out.shp before perm:', out.shape)
        out = out.permute(0, 3, 2, 1)
        # print('out.shp:', out.shape)
        # print('out:', out)
        out = out.reshape(batch_size, -1, self.gru_input_size)
        # print('out after resh:', out.shape)

        out, gru_h = self.gru(out, self.gru_h)
        # print('gru_h.shp:',gru_h.shape)
        self.gru_h = gru_h.detach()
        # print("out.shape: ", str(out.shape))
        # print("test: ", F.softmax(self.fc(out[0])))
        """print(
            "stack shape :",
            str(len([F.log_softmax(self.fc(out[i]), 1) for i in range(out.shape[0])])),
        )"""
        # print([F.log_softmax(self.fc(out[i]), 1) for i in range(out.shape[0])])
        out = torch.stack(
            [F.log_softmax(self.fc(out[i]), 1) for i in range(out.shape[0])]
        )
        # print("out.shape stacked: ", str(out.shape))
        # print('final out.shp:', out.shape)
        return out

    def reset_hidden(self, batch_size):
        h = torch.zeros(self.gru_num_layers * 2, batch_size, self.gru_hidden_size)
        self.gru_h = Variable(h)

    def simple_forward(self, x):
        out = self.cnn(x)
        return out.permute(0, 2, 3, 1).detach().numpy()


class CRNN_lstm(nn.Module):

    def __init__(self, num_classes):
        super(CRNN_lstm, self).__init__()

        self.num_classes = num_classes + 1
        self.image_H = 44

        self.cnn = simple_CNN()
        # http://layer-calc.com/
        # c= 64 h=10 w=43

        self.postconv_height = 7
        self.postconv_width = 35

        self.lstm_input_size = self.postconv_height * 64
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 2
        self.lstm_h = None
        self.lstm_c = None

        self.lstm = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(self.lstm_hidden_size * 2, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.cnn(x)

        out = out.permute(0, 3, 2, 1)
        out = out.reshape(batch_size, -1, self.lstm_input_size)

        out, (lstm_h, lstm_c) = self.lstm(out, (self.lstm_h, self.lstm_c))
        self.lstm_h = lstm_h.detach()
        self.lstm_c = lstm_c.detach()
        out = torch.stack(
            [F.log_softmax(self.fc(out[i]), 1) for i in range(out.shape[0])]
        )
        return out

    def reset_hidden(self, batch_size):
        h = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size)
        c = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size)
        self.lstm_h = Variable(h)
        self.lstm_c = Variable(c)

    def simple_forward(self, x):
        out = self.cnn(x)
        return out.permute(0, 2, 3, 1).detach().numpy()


class CRNN_rnn(nn.Module):

    def __init__(self, num_classes):
        super(CRNN_rnn, self).__init__()

        self.num_classes = num_classes + 1
        self.image_H = 44

        self.cnn = simple_CNN()
        # http://layer-calc.com/
        # c= 64 h=10 w=43

        self.postconv_height = 7
        self.postconv_width = 35

        self.rnn_input_size = self.postconv_height * 64
        self.rnn_hidden_size = 128
        self.rnn_num_layers = 2
        self.rnn_h = None
        self.rnn = nn.RNN(
            self.rnn_input_size,
            self.rnn_hidden_size,
            self.rnn_num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(self.rnn_hidden_size, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.cnn(x)

        out = out.permute(0, 3, 2, 1)
        out = out.reshape(batch_size, -1, self.rnn_input_size)

        out, rnn_h = self.rnn(out, self.rnn_h)
        # print(rnn_h.shape)
        # print(out.shape)

        self.rnn_h = rnn_h.detach()
        out = torch.stack(
            [F.log_softmax(self.fc(out[i]), 1) for i in range(out.shape[0])]
        )

        return out

    def reset_hidden(self, batch_size):
        h = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size)
        self.rnn_h = Variable(h)

    def simple_forward(self, x):
        out = self.cnn(x)
        return out.permute(0, 2, 3, 1).detach().numpy()


def visualize_model(loader, model):
    for batch_id, (x_test, y_test, _) in enumerate(loader):
        for j in range(len(x_test)):
            # img = x_test[j].unsqueeze(0)
            # img = img.unsqueeze(1)
            img = x_test[j].unsqueeze(0)
            # img = im.unsqueeze(1)
            h = img.type(torch.float32)
            model_history = tl.log_forward_pass(
                model, h, layers_to_save="all", vis_opt="rolled"
            )
            print(model_history)
    os.system("pause")


def conv_layer_plot(nrows, ncols, title, image, figsize=(14, 3), color="gray"):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(title)
    # fig.tight_layout()
    for i in range(nrows * ncols):
        image_plot = axs[i // ncols, i % ncols].imshow(image[0, :, :, i], cmap=color)
        axs[i // ncols, i % ncols].axis("off")
    fig.subplots_adjust(right=0.8, top=0.98, bottom=0.02, hspace=0.2)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(image_plot, cax=cbar_ax)
    plt.show()


def fdl_layer_plot(image, title, figsize=(16, 8)):
    fig, axs = plt.subplots(1, figsize=figsize)
    fig.suptitle(title)
    image_plot = axs.imshow(image, cmap="gray")
    fig.colorbar(image_plot)
    axs.axis("on")
    plt.show()


def visualize_featuremap(crnn, loader, number):
    for batch_id, (x_test, y_test, _) in enumerate(loader):
        # x_test = x_test.view(x_test.shape[0], 1, x_test.shape[2], x_test.shape[3])

        for j in range(len(x_test)):
            print(x_test[j].shape)
            im = x_test[j].permute(1, 2, 0)
            plt.imshow(im, cmap="gray")
            plt.show()
            img = x_test[j].unsqueeze(0)
            # img = im.unsqueeze(1)
            h = img.type(torch.float32)
            print(type(x_test[j]))
            out = crnn.simple_forward(h)
            print("out.shp:", out.shape)
            conv_layer_plot(nrows=16, ncols=4, title="", image=out)
            number -= 1
            if number <= 0:
                break

        if number <= 0:
            break


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x
