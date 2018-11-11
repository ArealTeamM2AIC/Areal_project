from PIL import Image
import re
from os import listdir
from os.path import isfile, join


def load_images_in_path(path):
    return {f: load_one_image(join(path, f)) for f in listdir(path) if isfile(join(path, f))}


def load_one_image(file):
    return Image.open(file)


def get_town_name_list(img_dict):
    names = [f for f,img in img_dict.items()]
    names = list(map(lambda n: re.search("[A-Za-z]+", n).group(0), names))
    return set(names)
