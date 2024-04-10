import torch.nn as nn
from torchvision.transforms import v2
import torch.utils.data as data_utils
from files.d2l import Animator, Accumulator, Timer
from files.data import read_words_generate_csv, read_bbox_csv_show_image, \
    get_dataloaders, dataloader_show, read_maps
from files.dataset import CustomObjectDetectionDataset
from files.transform import ResizeWithPad
import torch
from files.model import CRNN, visualize_model, visualize_featuremap
from files.model_bbox import TinySSD, multibox_target
from files.test_train import train
from files.functions import generated_data_dir, htr_ds_dir
from wakepy import keep

# Todo confusion matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_transform = v2.Compose(
    [ResizeWithPad(h=28, w=140),
     v2.Grayscale()
     ])
do = 1
text_label_max_length = 6

if do == 1:
    with keep.running() as k:
        print('training word reader')
        read_words_generate_csv()

        char_to_int_map, int_to_char_map, char_set = read_maps()
        print('char_set', char_set)
        #char_to_int_map['_'] = '15'
        #int_to_char_map['15'] = '_'
        int_to_char_map['16'] = ''

        trl, tl = get_dataloaders(image_transform, char_to_int_map, 1000, text_label_max_length, char_set)

        dataloader_show(trl, number_of_images=2, int_to_char_map=int_to_char_map)

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
        # https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction

if do == 2:
    print('visualize featuremap')
    char_to_int_map, int_to_char_map, char_set = read_maps()
    crnn = CRNN().to(device)
    crnn.load_state_dict(torch.load(generated_data_dir() + 'trained_reader'))
    trl, tl = get_dataloaders(image_transform, char_to_int_map, 5, text_label_max_length, char_set)
    visualize_featuremap(crnn, trl)

if do == 3:
    print('visualize model')
    char_to_int_map, int_to_char_map, char_set = read_maps()
    crnn = CRNN().to(device)
    crnn.load_state_dict(torch.load(generated_data_dir() + 'trained_reader'))
    trl, tl = get_dataloaders(image_transform, char_to_int_map, 5, text_label_max_length, char_set)
    visualize_model(trl, crnn)

if do == 4:
    with keep.running() as k:

        #read_bbox_csv_show_image()
        #sizes = [[0.2 * 256, 0.272 * 256], [0.37 * 256, 0.447 * 256], [0.54 * 256, 0.619 * 256], [0.71 * 256, 0.79 * 256], [0.88 * 256, 0.961 * 256]]
        sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
                 [0.88, 0.961]]
        ratios = [[1, 2, 0.5]] * 5
        num_anchors = len(sizes[0]) + len(ratios[0]) - 1

        net = TinySSD(num_classes=1, num_anchors=num_anchors)
        X = torch.zeros((1, 3, 1280, 1024))
        anchors, cls_preds, bbox_preds = net(X)

        print('output anchors:', anchors.shape)
        print('output class preds:', cls_preds.shape)
        print('output bbox preds:', bbox_preds.shape)

        annotations_file = htr_ds_dir() + 'train/' + '_annotations.csv'
        image_folder = htr_ds_dir() + 'train/'

        ds = CustomObjectDetectionDataset(annotations_file, image_folder, 5)
        print('len.ds:', len(ds))
        #indices = torch.arange(4)
        #ds_lim = data_utils.Subset(ds, indices)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

        trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

        cls_loss = nn.CrossEntropyLoss(reduction='none')
        bbox_loss = nn.L1Loss(reduction='none')


        def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
            print('cls_preds:', cls_preds.shape)
            print('cls_labels:', cls_labels.shape)
            print('bbox_preds:', bbox_preds.shape)
            print('bbox_labels:', bbox_labels.shape)
            #print('bbox_lbls', bbox_labels)
            #print('resh cls_preds:', cls_preds.reshape(-1, cls_preds.shape[2]))
            #print('resh cls_lbls:', cls_labels.reshape(-1)).reshape(cls_preds.shape[0], -1).mean(dim=1)

            batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
            cls = cls_loss(cls_preds.reshape(-1, num_classes),
                           cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
            bbox = bbox_loss(bbox_preds * bbox_masks,
                             bbox_labels * bbox_masks).mean(dim=1)
            return cls + bbox


        def cls_eval(cls_preds, cls_labels):
            # Because the class prediction results are on the final dimension,
            # `argmax` needs to specify this dimension
            return float((cls_preds.argmax(dim=-1).type(
                cls_labels.dtype) == cls_labels).sum())


        def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
            return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


        num_epochs = 5
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
        timer = Timer()
        train_iter = train_loader
        net = net.to(device)
        for epoch in range(num_epochs):
            # Sum of training accuracy, no. of examples in sum of training accuracy,
            # Sum of absolute error, no. of examples in sum of absolute error
            metric = Accumulator(4)
            net.train()
            for features, target in train_iter:
                trainer.zero_grad()
                X, Y = features.to(device), target.to(device)
                print('X.shp:', X.shape)
                print('X', X)
                print('Y.shp:', Y[0].shape)
                print('Y', Y[0, :, :])
                print("dimensions Y:", Y[0].ndimension())

                # Generate multiscale anchor boxes and predict their classes and
                # offsets
                anchors, cls_preds, bbox_preds = net(X)
                # Label the classes and offsets of these anchor boxes
                #label = Y[0, :, :]
                #print('label train shp:', label.shape)
                bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y[0, :, :, :])
                print('bbox_lbls:', bbox_labels)
                # Calculate the loss function using the predicted and labeled values
                # of the classes and offsets
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                              bbox_masks)
                l.mean().backward()
                trainer.step()
                metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                           bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                           bbox_labels.numel())
                print('training loop end')
            cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
            animator.add(epoch + 1, (cls_err, bbox_mae))
        print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
        print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
              f'{str(device)}')
        print(ds[0][0])

if do == 5:
    image_transform = v2.Compose(
        [
            v2.Grayscale()
        ])
    read_bbox_csv_show_image()
