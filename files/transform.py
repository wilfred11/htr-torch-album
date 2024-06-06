from torchvision.transforms import functional as F
from torchvision.transforms.v2.functional import get_size


class TextToInt:

    def __init__(self, char_to_int_map):
        self.char_map = char_to_int_map

    def __call__(self, text):
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(int(ch))
        return int_sequence


class IntToText:
    def __init__(self, int_to_char_map):
        self.int_map = int_to_char_map

    def __call__(self, integer_tensor):
        char_sequence = []
        for i in integer_tensor:
            if i == ' ':
                ch = self.int_map['']
            else:
                ch = self.int_map[str(i.item())]
            char_sequence.append(ch)
        return char_sequence


class IntToString:
    def __init__(self, int_to_char_map):
        self.int_map = int_to_char_map

    def __call__(self, integer_tensor):
        char_sequence = []
        for i in integer_tensor:
            if i == ' ':
                ch = self.int_map['']
            else:
                ch = self.int_map[str(i.item())]
            char_sequence.append(ch)
        string = "".join([str(c) for c in char_sequence])
        return string


class FillArray:

    def __init__(self, length):
        self.length = length

    def __call__(self, array):
        for i in range(self.length - len(array)):
            array.append(16)

        return array


class ResizeWithPad:
    def __init__(self, w=156, h=44):
        self.w = w
        self.h = h

    def __call__(self, image):
        sz = get_size(image)
        w_1 = sz[0]
        h_1 = sz[1]

        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1

        if round(ratio_1, 2) != round(ratio_f, 2):
            hp = int(w_1 / ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])

        else:
            return F.resize(image, [self.h, self.w])
