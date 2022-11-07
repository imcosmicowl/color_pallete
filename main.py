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
parser = argparse.ArgumentParser(description="Color palett extractor",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--path', type=str, help='path to img or folder with imgs', required=True)
parser.add_argument('--num_colors', '-nc', type=int, help='number of colors to be extracted', required=True)
parser.add_argument('--out_folder', type=str, help='output folder', required=True)
parser.add_argument('--store_pallet', '-sp', type=bool, help="yes/no to estore color pallet", required=False)

args = parser.parse_args()


def get_color_pallet(input_file: str, num_colors: int) -> list[tuple]:
    """
    Gets the most num_colors present colors from an image.

    :param input_file: str
            Image file to extract color pallet
    :param num_colors: int
            Number of colors to be extracted.
    :return: list[tuple]
            A list of tuple in the form (r, g, b)
    """
    color_thief = ColorThief(input_file)
    palette = color_thief.get_palette(color_count=num_colors)

    # DEBUG
    print(f"Color palette for {input_file}: {palette}")

    return palette


def append_color_pallet(original_image: str, color_pallet: list[tuple], output_file: str) -> None:
    """
    Appends the color pallet image at the bottom of the original image.

    :param original_image: str
            Image file.
    :param color_pallet: list[tuple]
            A list of tuple in the form (r, g, b)
    :param output_file: str
            Output file to store the new image.
    :return: list[tuple]
            A list of tuple in the form (r, g, b)
    """
    og_img = Image.open(original_image)
    og_width, og_height = og_img.size
    pallete_width, pallete_height, _ = color_pallet.shape

    height_offset = math.ceil(og_height / 20)
    if og_height > og_width:
        height_offset = math.ceil(og_height / 30)

    total_width = og_width
    total_height = og_height + pallete_height + (height_offset * 2)

    combined_img = Image.new('RGB', (total_width, total_height), (0, 0, 0))

    starting_point_to_paste = round((og_width - pallete_width) / 2)
    combined_img.paste(og_img, (0, 0))
    combined_img.paste(color_pallet, (starting_point_to_paste, og_height + height_offset))

    combined_img.save(output_file)


def pallet_to_img(pallet: list[tuple], output_file: str, input_img: str, store_pallet: bool) -> np.ndarray:
    """
        Creates an image from the list of color pallets.

        :param pallet: list[tuple]
                A list of tuple in the form (r, g, b)
        :param output_file: str
                Output file to store the new image
        :param input_img: str
                Input image file name.
        :param store_pallet: bool
                Boolean indicating whether to store the pallet as an image or not.
        :return: np.ndarray([n.m)]
                Two-dimensional array containing the color pallet appended to the image.
    """
    h, w, _ = cv2.imread(input_img).shape
    pallete_w = round(w * 0.95)
    pallete_h = round(h * 1 / 5)
    color_pallete_img = np.zeros((pallete_h, pallete_w, 3))
    square_w = round(pallete_w / args.num_colors)
    ii = 0
    for color in pallet:
        cv2.rectangle(color_pallete_img, (ii * square_w, 0), ((ii + 1) * square_w, pallete_h), tuple(reversed(color)),
                      -1)
        ii += 1
    if store_pallet:
        cv2.imwrite(output_file, color_pallete_img)
    return color_pallete_img


def create_pallet(filename: str, num_colors: int, abs_path: str, out_folder: str, store_pallet: bool) -> None:
    """
        Given an image file (filename), extract the (num_colors-colors) color pallet, appends it to the original image,
        and stores it at out_folder. Whether to store the color pallet image or not is determined by store_pallet.

        :param filename: str
                Input image path.
        :param num_colors: int
                Number of colors to be extracted.
        :param abs_path: str
                Absolute path of the folder containing the image.
        :param out_folder: str
                Output filename of the image.
        :param store_pallet: bool
                Boolean indicating whether to store the pallet as an image or not.
        :return: None

    """
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

    output_palette = file_prefix + file_split[0] + '_palett.' + file_split[1]
    output_combined = file_prefix + file_split[0] + '_with_palett.' + file_split[1]
    output_palette = join(out_folder, output_palette)
    output_combined = join(out_folder, output_combined)
    palette_list = get_color_pallet(join(abs_path, filename), num_colors)
    color_pallete_aux = pallet_to_img(palette_list, output_palette, join(abs_path, filename), store_pallet)
    append_color_pallet(join(abs_path, filename), color_pallete_aux, output_combined)


def get_hex_color(color: tuple[int, int, int]) -> str:
    """
    Converts RGB color values to hex.
    :param color: bool
            Color value as R,G,B.
    :return: str
            Hex color value as a string.
    """
    return '#%02x%02x%02x' % color


def main(path, num_colors, out_folder, store_pallete):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for file in onlyfiles:
        create_pallet(file, int(num_colors), path, out_folder, store_pallete)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main(args.path, args.num_colors, args.out_folder,
         True if args.store_pallete == 'yes' or args.store_pallete == 'y' else False)
