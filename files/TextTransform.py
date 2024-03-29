class TextToInt:
    """Maps characters to integers and vice versa"""

    def __init__(self, char_to_int_map):
        # self.char_map_str = char_map_str
        self.char_map = char_to_int_map
        # self.index_map = {}

    def __call__(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(int(ch))
        return int_sequence


class FillArray:
    """Maps characters to integers and vice versa"""

    def __init__(self, length):
        self.length = length

    def __call__(self, array):
        # empty_array_length = self.length - len(array)
        #print('lenarray:', len(array))
        #print('length:', self.length)
        for i in range(self.length - len(array)):
            #print('appending')
            array.append(69)

        #print('arrlength: ',len(array) )
        return array


class IntToText:
    """Maps characters to integers and vice versa"""

    def __init__(self, char_map_str):
        # self.char_map_str = char_map_str
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def __call__(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')
