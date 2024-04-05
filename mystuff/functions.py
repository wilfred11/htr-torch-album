import csv


def external_data_dir():
    return '../htr-torch-data/'


def generated_data_dir():
    return external_data_dir() + 'generated/'


def iam_dir():
    return external_data_dir() + 'iam/'


def htr_ds_dir():
    return external_data_dir() + 'htr-dataset/'


def ascii_dir():
    return iam_dir() + 'ascii/'


def words_file():
    return ascii_dir() + 'words.txt'


def read_maps():
    char_to_int_map = {}
    int_to_char_map = {}
    char_set = set()
    with open(external_data_dir() + 'char_map_15.csv', 'r') as file:
        csv_reader = csv.reader(file, delimiter=';', quotechar='\'', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for row in csv_reader:
            char_to_int_map[row[0]] = row[1]
            int_to_char_map[row[1]] = row[0]
            char_set.add(row[0])

    return char_to_int_map, int_to_char_map, char_set
