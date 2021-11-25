import os

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from scipy import cluster
import pandas as pd
import math
import colorsys
import argparse
from colorthief import ColorThief
import cv2

"""
Parser management
"""
parser = argparse.ArgumentParser(description="Color palette extractor",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--path', type=str, help='path to img or folder with imgs')
parser.add_argument('--num_colors', '-nc', type=int, help='number of colors to be extracted')
parser.add_argument('--text', '-t', default=False, required=False)

args = parser.parse_args()


# GLOBALS DEFFINITIONS


def get_color_pallete_(input_file, output_file, num_colors, display_color=False):
    img = plt.imread(input_file)

    red, green, blue = [], [], []
    for line in img:
        for pixel in line:
            r, g, b = pixel
            red.append(r)
            green.append(g)
            blue.append(b)

    df = pd.DataFrame({
        'red': red,
        'green': green,
        'blue': blue
    })

    df['standardized_red'] = cluster.vq.whiten(df['red'])
    df['standardized_green'] = cluster.vq.whiten(df['green'])
    df['standardized_blue'] = cluster.vq.whiten(df['blue'])

    color_pallete, distortion = cluster.vq.kmeans(df[['standardized_red', 'standardized_green', 'standardized_blue']],
                                                  num_colors)
    colors = []
    red_std, green_std, blue_std = df[['red', 'green', 'blue']].std()
    for color in color_pallete:
        scaled_red, scaled_green, scaled_blue = color
        colors.append((
            math.ceil(scaled_red * red_std),
            math.ceil(scaled_green * green_std),
            math.ceil(scaled_blue * blue_std)
        ))

    colors.sort(key=lambda x: step(x[0], x[1], x[2], 8))

    # FIXME: need a smart way to resize fonts based on picture size
    font_size = 11
    font = ImageFont.truetype("Roboto-Medium.ttf", font_size)
    sample_text = '#F8F8F7'
    proper_font_size = False

    pil_img = Image.open(input_file)
    pil_width, pil_height = pil_img.size
    height = 0
    if pil_height > pil_width:
        height = math.floor(pil_height / 6)
    else:
        height = math.floor(pil_height / 4)

    pallete = Image.new('RGB', (pil_width, height), (255, 255, 255))
    single_img_space = math.floor(pil_width / num_colors)
    single_img_offset = math.floor(single_img_space / 14)
    total_offset = single_img_offset * (num_colors + 1)
    single_img_width = math.floor((pil_width - total_offset) / num_colors)
    single_img_space = single_img_width + single_img_offset

    final_img_width = (single_img_width + (pil_width - (single_img_space * num_colors))) - single_img_offset

    while not proper_font_size:
        if get_text_width(font, sample_text) > single_img_width and font_size > 1:
            font_size -= 1
            font = ImageFont.truetype("Roboto-Medium.ttf", font_size)
        elif get_text_width(font, sample_text) < single_img_width - 20:
            font_size += 1
            font = ImageFont.truetype("Roboto-Medium.ttf", font_size)
        else:
            proper_font_size = True

    x_offset = 0
    for i in range(len(colors)):
        if i == len(colors) - 1:
            new_img = Image.new('RGB', (final_img_width, height), colors[i])
            pallete.paste(new_img, (x_offset, 0))
            if display_color:
                draw = ImageDraw.Draw(pallete)
                draw.text((x_offset, height - 20 - get_text_height(font, sample_text)), get_hex_color(colors[i]),
                          (255, 255, 255), font=font)
        elif i == 0:
            new_img = Image.new('RGB', (single_img_width, height), colors[i])
            pallete.paste(new_img, (single_img_offset, 0))
            if display_color:
                draw = ImageDraw.Draw(pallete)
                draw.text((single_img_offset, height - 20 - get_text_height(font, sample_text)),
                          get_hex_color(colors[i]), (255, 255, 255), font=font)
            x_offset += single_img_space + single_img_offset
        else:
            new_img = Image.new('RGB', (single_img_width, height), colors[i])
            pallete.paste(new_img, (x_offset, 0))
            if display_color:
                draw = ImageDraw.Draw(pallete)
                draw.text((x_offset, height - 20 - get_text_height(font, sample_text)), get_hex_color(colors[i]),
                          (255, 255, 255), font=font)
            x_offset += single_img_space

    pallete.save(output_file)


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

    combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    combined_img.paste(og_img, (0, 0))
    combined_img.paste(pallete_img, (0, og_height + height_offset))

    combined_img.save(output_file)


def pallete_to_img(palette, output_file, input_img):

    h, w, _ = cv2.imread(input_img).shape
    pallete_w = round(w*0.95)
    pallete_h = round(h*1/5)
    color_pallete_img = np.zeros((pallete_h,pallete_w,3))
    square_w = round(pallete_w/args.num_colors)
    ii = 0
    for color in palette:
        cv2.rectangle(color_pallete_img, (ii*square_w,0), ((ii+1)*square_w, pallete_h), tuple(reversed(color)), -1)
        ii += 1

    cv2.imwrite(output_file, color_pallete_img)

def create_pallete(filename, num_colors, display_color=False):
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
    palette_list = get_color_pallete(filename, num_colors)
    pallete_to_img(palette_list, output_palette, filename)
    append_color_pallete(filename, output_palette, output_combined)


def step(r, g, b, repititions=1):
    lum = math.sqrt(0.241 * r + 0.691 * g + 0.068 * b)

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    h2 = int(h * repititions)
    lum2 = int(lum * repititions)
    v2 = int(v * repititions)

    if h2 % 2 == 1:
        v2 = repititions - v2
        lum = repititions - lum

    return (h2, lum, v2)


def get_hex_color(color):
    return '#%02x%02x%02x' % color


def get_text_width(font, text):
    width = 0
    for ch in text:
        width += font.getsize(ch)[0]
    return widthlum2


def get_text_height(font, text):
    height = []
    for ch in text:
        height.append(font.getsize(ch)[1])
    return max(height)


def main(path, num_colors, text):
    create_pallete(path, int(num_colors), text)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main(args.path, args.num_colors, args.text)
