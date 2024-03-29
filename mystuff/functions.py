
def external_data_dir():
    return '../htr-torch-data/'


def generated_data_dir():
    return external_data_dir() + 'generated/'


def iam_dir():
    return external_data_dir() + 'iam/'


def ascii_dir():
    return iam_dir() + 'ascii/'


def words_file():
    return ascii_dir() + 'words.txt'

