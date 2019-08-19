import numpy as np
from skimage.measure import label
from tifffile import imsave as tifsave
from imageio import imread
import argparse
import glob
import os

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir_cytoplasm", required=True, help="path to folder containing images")
parser.add_argument("--input_dir_overlap", required=True, help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")

a = parser.parse_args()

# def convert_files_to_array(files):
#
#     resulting_list = []
#
#     for file in files:
#         file_as_array = imread(file)
#         resulting_list.append(file_as_array)
#
#     resulting_array = np.asarray(resulting_list)
#
#     print(resulting_array[0, :, :, 0])
#     return resulting_array[:, :, :, 0]


def convert_files_to_array(files):

    resulting_list = []

    for file in files:
        file_as_array = imread(file)
        resulting_list.append(file_as_array)

    resulting_array = np.asarray(resulting_list)
    resulting_array[resulting_array >= 127] = 255
    resulting_array[resulting_array < 127] = 0

    return resulting_array


def interleave_cytoplasm_overlap(cytoplasm, overlap):

    try:
        overlap = overlap[:, :, :, 0]
    except:
        pass

    try:
        cytoplasm = cytoplasm[:, :, :, 0]
    except:
        pass

    zero_array = np.zeros((512, 512))
    overlap = np.append(overlap, zero_array)
    overlap = overlap.reshape((100, 512, 512))

    interleaved_array = np.zeros((100, 512, 512, 2))
    interleaved_array[:, :, :, 0] = cytoplasm
    interleaved_array[:, :, :, 1] = overlap

    stack = interleaved_array.swapaxes(1, 3).swapaxes(2, 3)
    stack = np.resize(stack, (2 * 100, 512, 512))
    return stack


def relabel(stack):

    relabeled = label(stack, connectivity=1)
    relabeled = relabeled[0::2]
    relabeled = relabeled.astype(np.int16)
    return relabeled


def save(output, relabeled):
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    os.chdir(output)
    tifsave('truth.tif', relabeled) ##todo
    print('truth.tif created')


def main():

    input_cytoplasm = a.input_dir_cytoplasm
    input_overlap = a.input_dir_overlap
    output = a.output_dir

    all_files_cytoplasm = glob.glob(os.path.join(input_cytoplasm, '*.png'))
    all_files_overlap = glob.glob(os.path.join(input_overlap, '*.png'))

    cytoplasm = convert_files_to_array(all_files_cytoplasm)
    overlap = convert_files_to_array(all_files_overlap)

    stack = interleave_cytoplasm_overlap(cytoplasm, overlap)
    relabeled = relabel(stack)
    save(output, relabeled)


main()
