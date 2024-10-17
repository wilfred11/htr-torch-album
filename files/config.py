import csv


class Config:
    def __init__(self, char_file, text_label_max_length):
        self.char_file = char_file
        maps = self.read_maps(char_file)
        self.int_to_char_map = maps[1]
        self.char_to_int_map = maps[0]
        self.char_set = maps[2]
        self.empty_label = maps[3]
        self.text_label_max_length = text_label_max_length
        self.num_epoch = 5
        self.blank_label = self.get_blank_label()

    def read_maps(self, char_file):
        char_to_int_map = {}
        int_to_char_map = {}
        char_set = set()
        with open(char_file, "r") as file:
            csv_reader = csv.reader(
                file,
                delimiter=";",
                quotechar="'",
                quoting=csv.QUOTE_MINIMAL,
                lineterminator="\n",
            )
            for row in csv_reader:
                char_to_int_map[row[0]] = row[1]
                int_to_char_map[row[1]] = row[0]
                char_set.add(row[0])

        empty_label = len(char_set)
        self.num_classes = empty_label
        int_to_char_map[str(empty_label)] = ""

        self.char_to_int_map = char_to_int_map
        self.int_to_char_map = int_to_char_map
        self.char_set = char_set
        return char_to_int_map, int_to_char_map, char_set, empty_label

    def get_blank_label(self):
        return int(self.char_to_int_map.get("_"))
