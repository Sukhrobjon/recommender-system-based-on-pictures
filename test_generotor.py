import shutil
import os


def read_img_ids_from_file(filename):
    """
    reads the the image ids from preselected 2000 image ids
    """
    img_ids = []
    with open(filename) as file:
        for line in file:
            line = line.strip().split("\n")
            line = "".join(str(char) for char in line)
            img_ids.append(line)

    return img_ids


filename = 'smaller_chunk_of_samples.txt'
to_file = '../images_2000_product'
files = read_img_ids_from_file(filename)

for imgs in files:
    shutil.copy(imgs, to_file)


