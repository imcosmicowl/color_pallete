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

parser.add_argument('--path', type=str, help='path to img or folder with imgs', required=True)
parser.add_argument('--num_colors', '-nc', type=int, help='number of colors to be extracted', required=True)
parser.add_argument('--out_folder', type=str, help='output folder', required=True)
parser.add_argument('--store_pallete', '-sp', type=bool, help="yes/no to estore color pallete", required=False)

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
    pallete_width, pallete_height,_ = color_pallete.shape

    height_offset = math.ceil(og_height / 20)
    if og_height > og_width:
        height_offset = math.ceil(og_height / 30)

    total_width = og_width
    total_height = og_height + pallete_height + (height_offset * 2)

    combined_img = Image.new('RGB', (total_width, total_height), (0, 0, 0))

    starting_point_to_paste = round((og_width - pallete_width)/2)
    combined_img.paste(og_img, (0, 0))
    combined_img.paste(color_pallete, (starting_point_to_paste, og_height + height_offset))

    combined_img.save(output_file)


def pallete_to_img(palette, output_file, input_img, store_pallete):
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
    if store_pallete:
        cv2.imwrite(output_file, color_pallete_img)
    return color_pallete_img


def create_pallete(filename, num_colors, abs_path, out_folder, store_pallete):
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
    output_palette = join(out_folder, output_palette)
    output_combined = join(out_folder, output_combined)
    palette_list = get_color_pallete(join(abs_path, filename), num_colors)
    color_pallete_aux = pallete_to_img(palette_list, output_palette, join(abs_path, filename), store_pallete)
    append_color_pallete(join(abs_path, filename), color_pallete_aux, output_combined)


def get_hex_color(color):
    return '#%02x%02x%02x' % color


def main(path, num_colors, out_folder, store_pallete):

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for file in onlyfiles:
        create_pallete(file, int(num_colors), path, out_folder, store_pallete)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main(args.path, args.num_colors, args.out_folder, True if args.store_pallete == 'yes' or args.store_pallete == 'y' else False)