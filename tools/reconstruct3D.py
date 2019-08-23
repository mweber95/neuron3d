import numpy as np
from skimage.measure import label
from tifffile import imsave as tifsave
from imageio import imread
import argparse
import glob
import os

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir_cytoplasm", required=True, help="path to folder containing cytoplasm images for relabeling")
parser.add_argument("--input_dir_overlap", required=True, help="path to folder containing overlap images for relabeling")
parser.add_argument("--output_dir", required=True, help="output directory")
parser.add_argument("--name", required=True, help="name of resulting tif file")

a = parser.parse_args()


def convert_files_to_array(files):

    resulting_list = []

    for file in files:
        file_as_array = imread(file)
        resulting_list.append(file_as_array)

    resulting_array = np.asarray(resulting_list)

    'binarizing images'
    resulting_array[resulting_array >= 127] = 255
    resulting_array[resulting_array < 127] = 0

    return resulting_array


def interleave_cytoplasm_overlap(cytoplasm, overlap):

    'checking right format for overlap images'
    try:
        overlap = overlap[:, :, :, 0]
    except:
        pass

    'checking right format for cytoplasm images'
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

    'connected components'
    relabeled = label(stack, connectivity=1)
    'discard overlap images'
    relabeled = relabeled[0::2]
    relabeled = relabeled.astype(np.int16)
    return relabeled


def save(output, relabeled, name):
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    os.chdir(output)
    tifsave(str(name + '.tif'), relabeled)
    print(str(name) + '.tif created')


def main():

    'argparse arguments'
    input_cytoplasm = a.input_dir_cytoplasm
    input_overlap = a.input_dir_overlap
    output = a.output_dir
    name = a.name

    'all images to variable'
    all_files_cytoplasm = glob.glob(os.path.join(input_cytoplasm, '*.png'))
    all_files_overlap = glob.glob(os.path.join(input_overlap, '*.png'))

    'converting files to array'
    cytoplasm = convert_files_to_array(all_files_cytoplasm)
    overlap = convert_files_to_array(all_files_overlap)

    'interleaved composition of image stack'
    stack = interleave_cytoplasm_overlap(cytoplasm, overlap)

    'relabeling with connected components'
    relabeled = relabel(stack)

    'saving tif'
    save(output, relabeled, name)


main()
