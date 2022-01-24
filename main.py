import argparse
import math
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from PIL import Image
from colorthief import ColorThief

"""
Parser management
"""
parser = argparse.ArgumentParser(description="Color palette extractor",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--path', type=str, help='path to img or folder with imgs')
parser.add_argument('--num_colors', '-nc', type=int, help='number of colors to be extracted')

args = parser.parse_args()


# GLOBALS DEFFINITIONS

def get_color_pallete(input_file, num_colors):
    color_thief = ColorThief(input_file)
    palette = color_thief.get_palette(color_count=num_colors)

    # DEBUG
    print(f"Color palette for {input_file}: {palette}")

    return palette


def append_color_pallete(original_image, color_pallete, output_file):
    og_img = Image.open(original_image)
    og_width, og_height = og_img.size
    pallete_img = Image.open(color_pallete)
    pallete_width, pallete_height = pallete_img.size

    height_offset = math.ceil(og_height / 20)
    if og_height > og_width:
        height_offset = math.ceil(og_height / 30)

    total_width = og_width
    total_height = og_height + pallete_height + (height_offset * 2)

    combined_img = Image.new('RGB', (total_width, total_height), (0, 0, 0))

    starting_point_to_paste = round((og_width - pallete_width)/2)
    combined_img.paste(og_img, (0, 0))
    combined_img.paste(pallete_img, (starting_point_to_paste, og_height + height_offset))

    combined_img.save(output_file)


def pallete_to_img(palette, output_file, input_img):
    h, w, _ = cv2.imread(input_img).shape
    pallete_w = round(w * 0.95)
    pallete_h = round(h * 1 / 5)
    color_pallete_img = np.zeros((pallete_h, pallete_w, 3))
    square_w = round(pallete_w / args.num_colors)
    ii = 0
    for color in palette:
        cv2.rectangle(color_pallete_img, (ii * square_w, 0), ((ii + 1) * square_w, pallete_h), tuple(reversed(color)),
                      -1)
        ii += 1

    cv2.imwrite(output_file, color_pallete_img)


def create_pallete(filename, num_colors, abs_path):
    file_path = filename.split('/')
    file_prefix = ''
    file_split = ''
    for i in range(len(file_path)):
        if i != len(file_path) - 1:
            file_prefix = file_prefix + file_path[i] + '/'
        else:
            file_split = file_path[i]
    file_split = file_split.split('.')
    # if file_split[1] != 'jpg' and file_split[1] != 'png':
    #     raise ("The file must be a jpg or png")

    output_palette = file_prefix + file_split[0] + '_palette.' + file_split[1]
    output_combined = file_prefix + file_split[0] + '_with_palette.' + file_split[1]
    palette_list = get_color_pallete(join(abs_path, filename), num_colors)
    pallete_to_img(palette_list, output_palette, join(abs_path, filename))
    append_color_pallete(join(abs_path, filename), output_palette, output_combined)


def get_hex_color(color):
    return '#%02x%02x%02x' % color


def main(path, num_colors):

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for file in onlyfiles:
        create_pallete(file, int(num_colors), path)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    # TODO store the output to desired folder, allow to select whether to store pallet or not
    main(args.path, args.num_colors)