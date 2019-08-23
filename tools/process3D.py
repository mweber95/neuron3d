import argparse
import os
import numpy as np
from tifffile import imread as tifread
from tifffile import imsave as tifsave
from PIL import Image
from scipy import ndimage
import glob
from imageio import imread
from scipy.ndimage.morphology import binary_erosion
from skimage.morphology import skeletonize

parser = argparse.ArgumentParser()
parser.add_argument("--operation", required=True, help="operation to be executed",
                    choices=["split", "cytoplasm", "membrane", "overlap", "consecutive", "eroskele",
                             "tif_to_png", "png_to_tif"])
parser.add_argument("--input_dir", required=True, help="path to folder containing tif file")
parser.add_argument("--output_dir", required=True, help="path to output folder")
parser.add_argument("--split", required=False, choices=["em", "label"], help="select em or label stack")
parser.add_argument("--size", required=False, default=512, help="image width and height")
parser.add_argument("--name", required=False, help="name for the output tif file")

a = parser.parse_args()


def split_snemi3d(input, output, prefix):

    stack = tifread(input)
    os.chdir(output)

    'slicing original image stack'
    stack_split_a = stack[:, 512:, 512:]
    stack_split_b = stack[:, :512, 512:]
    stack_split_c = stack[:, :512, :512]
    stack_split_d = stack[:, 512:, :512]

    'saving all four split image stacks'
    tifsave('%s_a.tif' % prefix, stack_split_a)
    tifsave('%s_b.tif' % prefix, stack_split_b)
    tifsave('%s_c.tif' % prefix, stack_split_c)
    tifsave('%s_d.tif' % prefix, stack_split_d)


def edges(slice):
    """
    - applying min and max filter on image slices to detect membranes
    - membranes are objects that have different values in a 3x3 window
    --> if min and max function are different, the pixel equals a membrane
    - this function "edges" is needed for creating overlap images, cytoplasma images, membrane_single_pictures and
    membrane_tif_stack
    :param slice: one slice from image stack
    :return: boolean array with membranes labeled true
    """
    edges_min = ndimage.generic_filter(slice, function=min, size=(3, 3))
    edges_max = ndimage.generic_filter(slice, function=max, size=(3, 3))
    edges_compared = edges_min != edges_max
    return edges_compared


def cytoplasm(input, output):

    stack = tifread(input)
    os.chdir(output)

    for i, slice in enumerate(stack):
        edges_compared = edges(slice)
        cytoplasm_logical = np.logical_and(np.logical_not(edges_compared), slice > 0)
        pictures = Image.fromarray(cytoplasm_logical.astype('uint8') * 255)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(stack)) + " files created")


def membrane(input, output):

    stack = tifread(input)
    os.chdir(output)

    for i, slice in enumerate(stack):
        edges_compared = edges(slice)
        pictures = Image.fromarray(edges_compared.astype('uint8') * 255)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(stack)) + " files created")


def overlap(input, output):

    stack = tifread(input)
    os.chdir(output)
    list_for_slices = []

    for i, slice in enumerate(stack):
        edges_compared = edges(slice)
        list_for_slices.append(np.invert(edges_compared))
        print(str(i + 1) + "/" + str(len(stack)) + " membrane pictures for overlap detection processed")

    membranes = np.asarray(list_for_slices)
    membranes_indices = membranes * stack

    stack1 = membranes_indices[:-1]
    stack2 = membranes_indices[1:]
    compared_stack = np.logical_and(np.logical_and(stack1 == stack2, stack1 > 0), stack2 > 0)

    for i, slice in enumerate(compared_stack):
        overlap_image = slice.astype('uint8') * 255
        pictures = Image.fromarray(overlap_image)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(compared_stack)) + " files created")


def consecutive(input, output, size):

    stack = tifread(input)
    os.chdir(output)

    if len(stack.shape) == 4:
        stack = stack[:, :, :, 0]

    stack1 = stack[:-1]
    stack2 = stack[1:]

    for i, (slice1, slice2) in enumerate(zip(stack1, stack2)):
        zero_array = np.zeros((3, int(size), int(size)))
        zero_array[0] = slice1
        zero_array[1] = slice2
        zero_array = np.ascontiguousarray(zero_array.transpose(1, 2, 0))
        consecutive_image = zero_array.astype('uint8')
        pictures = Image.fromarray(consecutive_image, 'RGB')
        pictures.save('%02d.png' % i)
        zero_array[0] = 0
        zero_array[1] = 0
        print(str(i + 1) + "/" + str(len(stack1)) + " files created")


def eroskele(input, output):

    list_files = []

    for file in input:
        file_as_array = imread(file)
        file_as_array[file_as_array < 127] = 0
        file_as_array[file_as_array >= 127] = 1
        try:
            file_as_array = file_as_array[:, :, 0]
        except:
            file_as_array = file_as_array[:, :]
        list_files.append(file_as_array)

    resulting_array = np.asarray(list_files)

    os.chdir(output)

    for i, slice in enumerate(resulting_array):

        slice = binary_erosion(slice)   #1
        slice = binary_erosion(slice)   #2
        slice = binary_erosion(slice)   #3
        slice = binary_erosion(slice)   #4
        slice = binary_erosion(slice)   #5
        slice = binary_erosion(slice)   #6
        slice = binary_erosion(slice)   #7
        slice = binary_erosion(slice)   #8
        slice = binary_erosion(slice)   #9
        slice = binary_erosion(slice)   #10
        slice = skeletonize(slice)

        pictures = Image.fromarray(slice.astype('uint8') * 255)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(resulting_array)) + " files created")


def tif_to_png(input, output):

    stack = tifread(input)
    os.chdir(output)

    for i, slice in enumerate(stack):
        pictures = Image.fromarray(slice)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(stack)) + " files created")


def png_to_tif(input, output, name):

    image_names = []
    image_read = []

    for file in input:
        image_names.append(file)
    for image in image_names:
        image_read.append(imread(image))

    array = np.asarray(image_read)
    #try:
    #    array_dim_red = array[:, :, :, 0]
    #except:
    #    array_dim_red = array[:, :, :]

    os.chdir(output)
    tifsave(str(name + '.tif'), array)
    print(str(name) + '.tif created')


def main():

    output = a.output_dir

    'creating folder if new output path is selected'
    if not os.path.exists(output):
        os.makedirs(output)

    'splitting the original dataset in four parts with same size'
    if a.operation == 'split':
        prefix = a.split
        input = a.input_dir
        if prefix == 'None':
            raise NameError('no prefix selected, please use whether "em" or "label" to specify your intention')
        split_snemi3d(input, output, prefix)

    'create cytoplasm images'
    if a.operation == 'cytoplasm':
        input = a.input_dir
        cytoplasm(input, output)

    'create membrane images'
    if a.operation == 'membrane':
        input = a.input_dir
        membrane(input, output)

    'create overlap images'
    if a.operation == 'overlap':
        input = a.input_dir
        overlap(input, output)

    'create consecutive images'
    if a.operation == 'consecutive':
        input = a.input_dir
        consecutive(input, output, a.size)

    'tenfold erosion and following skeletonization of overlap images'
    if a.operation == 'eroskele':
        input = glob.glob(os.path.join(a.input_dir, '*.png'))
        eroskele(input, output)

    'converting a tif to pngs'
    if a.operation == 'tif_to_png':
        input = a.input_dir
        tif_to_png(input, output)

    'converting pngs to a tif'
    if a.operation == 'png_to_tif':
        input = glob.glob(a.input_dir + '*outputs.png')
        png_to_tif(input, output, a.name)


main()
