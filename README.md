This application allows you to train and test a htr-model on words in the IAM dataset. It contains three torch
models one LSTM model, one GRU model and a plain RNN model. There is possibility to compare the errorrates of these
three models. As the application is set up it allows for training on data limited to 6-character words, containing
characters a up to o. There is a possibility to use some image augmentations. Also it is possible to train and test
multiple
models after one another. Multiple metrics are being produced wer, cer, after every epoch a csv-file is produced with
the test
attempts and the actual words. A csv-file is created containing the training words and the frequence they were being
used to
train the model. I have also created a epoch-iterator to be sure to train on new images all the time. Further more a
config file is present to
create the dictionaries for integer to character translation and the other way around.

For my application, I have put the data in a separate directory named htr-torch next to my app directory.

htr-torch-data
->generated
->iam

In the iam directory I have put the IAM dataset. The generated directory contains the intermediate files and pickles.

Possible ideas still to investigate:
https://kiran-prajapati.medium.com/hand-digit-recognition-using-recurrent-neural-network-in-pytorch-b8db24540537

https://medium.com/@mohini.1893/handwriting-text-recognition-236b33c5caa4

https://github.com/Mohini1893/Handwriting-Text-Recognition/blob/master/Initial%20Approach%202/CNN%20with%20Tensorflow%20on%20a%20character-only%20dataset.ipynb

https://www.youtube.com/watch?v=GxtMbmv169o

https://deepayan137.github.io/blog/markdown/2020/08/29/building-ocr.html

https://github.com/furqan4545/handwritten_text_detection_and_recognition/blob/master/handwritten_textDetectionV1.ipynb

https://www.youtube.com/watch?v=ZiUEdS_5Byc&t=857s

https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc

https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/

line segmentation

https://towardsdatascience.com/train-a-lines-segmentation-model-using-pytorch-34d4adab8296

https://blog.paperspace.com/object-localization-pytorch-2/

https://towardsdatascience.com/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0

https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction