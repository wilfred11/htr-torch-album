def external_data_dir():
    return "../htr-torch-data/"


def generated_data_dir():
    return external_data_dir() + "generated/"


def score_dir():
    return "scores/"


def base_no_aug_score_dir():
    return "scores/base/no_aug/"


def adv_aug_score_dir():
    return "scores/adv/aug/"


def no_aug_graphs():
    return "no_aug/graphs/"


def aug_graphs():
    return "aug/graphs/"


def base_no_aug_score_dir():
    return "scores/base/no_aug/"


def adv_no_aug_score_dir():
    return "scores/adv/no_aug/"


def base_aug_score_dir():
    return "scores/base/aug/"


def adv_aug_score_dir():
    return "scores/adv/aug/"


def iam_dir():
    return external_data_dir() + "iam/"


def htr_ds_dir():
    return external_data_dir() + "htr-dataset/"


def ascii_dir():
    return iam_dir() + "ascii/"


def words_file():
    return ascii_dir() + "words.txt"
