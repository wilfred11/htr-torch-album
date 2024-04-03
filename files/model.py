import os
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchview import draw_graph
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator, AnchorGenerator


class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()

        self.num_classes = (16 + 1)
        self.image_H = 28

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

        self.postconv_height = 3
        self.postconv_width = 31

        self.gru_input_size = self.postconv_height * 64
        self.gru_hidden_size = 128
        self.gru_num_layers = 2
        self.gru_h = None
        self.gru_cell = None

        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size, self.gru_num_layers, batch_first=True,
                          bidirectional=True)

        self.fc = nn.Linear(self.gru_hidden_size * 2, self.num_classes)

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

        out = out.permute(0, 3, 2, 1)
        out = out.reshape(batch_size, -1, self.gru_input_size)

        out, gru_h = self.gru(out, self.gru_h)
        self.gru_h = gru_h.detach()
        out = torch.stack([F.log_softmax(self.fc(out[i])) for i in range(out.shape[0])])

        return out

    def reset_hidden(self, batch_size):
        h = torch.zeros(self.gru_num_layers * 2, batch_size, self.gru_hidden_size)
        self.gru_h = Variable(h)


class BBox(nn.Module):
    def __init__(self):
        super(BBox, self).__init__()

        self.num_classes = (16 + 1)
        self.image_H = 3542

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

        self.backbone = nn.Sequential(self.conv1, self.in1, self.conv2, self.in2, self.conv3, self.in3, self.conv4,
                                      self.in4, self.conv5, self.in5, self.conv6, self.in6)
        # backbone.
        # anchor_generator = DefaultBoxGenerator(
        #    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        # )
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))

        self.postconv_height = 386
        self.postconv_width = 268

        self.bb_input_size = self.postconv_width * self.postconv_height * 64

        model = FasterRCNN(backbone=self.backbone, num_classes=2, out_channels=4)

        self.bb_fc1 = nn.Linear(self.bb_input_size, self.bb_input_size / 2)
        self.bb_fc2 = nn.Linear(self.bb_input_size / 2, self.bb_input_size / 4)
        self.bb_fc3 = nn.Linear(self.bb_input_size / 4, 4)

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

        box_t = self.box_fc1(out)
        box_t = F.relu(box_t)
        box_t = self.box_fc2(box_t)
        box_t = F.relu(box_t)
        box_t = self.box_fc3(box_t)
        box_t = F.relu(box_t)
        box_t = F.sigmoid(box_t)
        return box_t


def visualize_model(loader, model, transform):
    # batch = next(iter(loader))
    # print('btchshp:', batch[0].shape)
    # t_im= transform(batch[0])
    # print('t_imshp:', t_im)
    # yhat = model(t_im)

    # model_graph = draw_graph(model, input_data=batch[0])

    # model_graph.visual_graph

    # tag_scores = model(x_train)
    # print('tag_scores',tag_scores)
    # make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    model_graph = draw_graph(model, input_size=(1, 1, 28, 140), expand_nested=True)
    model_graph.visual_graph
    model_graph.resize_graph(scale=5.0)  # scale as per the view model_graph.visual_graph.render(format='svg')
    model_graph.visual_graph.render(format='png')
    os.system('pause')


def conv_max_layer_plot(nrows, ncols, title, image, figsize=(14, 3), color='gray'):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(title)
    fig.tight_layout()
    for i in range(nrows * ncols):
        image_plot = axs[i // 8, i % 8].imshow(image[0, :, :, i], cmap=color)
        axs[i // 8, i % 8].axis('off')
    fig.subplots_adjust(right=0.8, top=0.98, bottom=0.02, hspace=0)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(image_plot, cax=cbar_ax)
    plt.show()


def fdl_layer_plot(image, title, figsize=(16, 8)):
    fig, axs = plt.subplots(1, figsize=figsize)
    fig.suptitle(title)
    image_plot = axs.imshow(image, cmap='gray')
    fig.colorbar(image_plot)
    axs.axis('on')
    plt.show()


def visualize_featuremap(crnn, loader):
    print('vis fm')
    # batch = next(iter(loader))
    # conv_output = crnn.conv1(batch[0])
    # print('conv_:',conv_output)
    for batch_id, (x_test, y_test) in enumerate(loader):
        for j in range(len(x_test)):
            plt.imshow(x_test[j], cmap='gray')
            plt.show()
            print('im_shp:', x_test[j].shape)
            img = x_test[j].unsqueeze(0)
            img = img.unsqueeze(1)
            conv1_output = crnn.conv1(img)
            conv_output_image = conv1_output.permute(0, 2, 3, 1).detach().numpy()
            conv_max_layer_plot(nrows=4, ncols=8, title='First Conv2D', image=conv_output_image)
            os.system('pause')

            in1_output = crnn.in1(conv1_output)

            self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))

