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
                    choices=["create_em", "create_overlap", "create_consecutive", "create_membrane",
                             "create_membrane_tif",  "create_cytoplasma", "preprocessing", "split",
                             "preprocessing_membrane", "create_rgb", "dilation", "cutten", "cutten2", "single",
                             "erosion"])
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")
parser.add_argument("--split", choices=["em", "label"], help="select em or label stack")

a = parser.parse_args()


def erosion():
    all_files = glob.glob(os.path.join(a.input_dir, '*.png'))
    list_files = []

    print(all_files)
    for file in all_files:
        file_as_array = imread(file)
        file_as_array[file_as_array < 127] = 0
        file_as_array[file_as_array >= 127] = 1
        try:
            file_as_array = file_as_array[:, :, 0]
        except:
            file_as_array = file_as_array[:, :]
        list_files.append(file_as_array)

    resulting_array = np.asarray(list_files)
    os.chdir(a.output_dir)
    for i, slice in enumerate(resulting_array):
        slice = binary_erosion(slice)
        slice = binary_erosion(slice)
        slice = binary_erosion(slice)
        slice = binary_erosion(slice)
        slice = binary_erosion(slice)
        slice = binary_erosion(slice)
        slice = binary_erosion(slice)
        slice = binary_erosion(slice)
        slice = binary_erosion(slice)
        slice = binary_erosion(slice)
        slice = skeletonize(slice)
        pictures = Image.fromarray(slice.astype('uint8') * 255)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(resulting_array)) + " files created")


def single():

    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)

    for i, slice in enumerate(stack):
        pictures = Image.fromarray(slice.astype('uint8'))
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(stack)) + " files created")


def cutten():

    input = a.input_dir
    all_files = glob.glob(os.path.join(input, '*.png'))
    list_files = []
    print(all_files)
    for file in all_files:
        file_as_array = imread(file)
        list_files.append(file_as_array)

    resulting_array = np.asarray(list_files)
    os.chdir(a.output_dir)
    tifsave('cyto_a.tif', resulting_array)


def cutten2():

    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)
    #stack = stack[:, 1:, 1:]
    #stack = stack[:, :-1, 1:]
    #stack = stack[:, :-1, :-1]
    stack = stack[:, 1:, :-1]

    tifsave('overlap_d.tif', stack)


def split_snemi3d(prefix):

    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)
    stack_split_c = stack[:, :512, :512]
    stack_split_b = stack[:, :512, 512:]
    stack_split_d = stack[:, 512:, :512]
    stack_split_a = stack[:, 512:, 512:]
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
    :return: boolean array with membranes labeled True
    """
    edges_min = ndimage.generic_filter(slice, function=min, size=(3, 3))
    edges_max = ndimage.generic_filter(slice, function=max, size=(3, 3))
    edges_compared = edges_min != edges_max
    return edges_compared


def membrane_single_pictures2():
    """
    - creating single EM pictures from train-input.tif for cytoplasma training
    """
    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)

    for i, slice in enumerate(stack):
        pictures = Image.fromarray(slice.astype('uint8'))
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(stack)) + " files created")


def em():
    """
    - creating single EM pictures from train-input.tif for cytoplasma training
    """
    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)

    for i, slice in enumerate(stack):
        pictures = Image.fromarray(slice)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(stack)) + " files created")


def overlap():
    """
    - creating single overlap images from train-labels.tif for overlap training
    - to ensure exact distinction and segmentation of overlap labels, the membrane of each object needs to be subtracted
    from the overlap
    """
    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)
    list_for_slices = []

    for i, slice in enumerate(stack):
        edges_compared = edges(slice)
        list_for_slices.append(np.invert(edges_compared))
        print(str(i + 1) + "/" + str(len(stack)) + " membrane pictures for overlap detection processed")

    membranes = np.asarray(list_for_slices)
    new_stack = membranes * stack

    stack1 = new_stack[:-1]
    stack2 = new_stack[1:]
    compared_stack = np.logical_and(np.logical_and(stack1 == stack2, stack1 > 0), stack2 > 0)

    for i, slice in enumerate(compared_stack):
        overlap_image = slice.astype('uint8') * 255
        pictures = Image.fromarray(overlap_image)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(compared_stack)) + " files created")


def consecutive():
    """
    - consecutive images are required for training with overlap images
    - picture i and picture i+1 are added to the red and green channel
    - transpose function is needed for the correct 3D arrangement of the .tif image stack
    """
    stack = tifread(a.input_dir)
    try:
        stack = stack[:, :, :, 0]
    except:
        stack = stack[:, :, :]
    os.chdir(a.output_dir)
    stack1 = stack[:-1]
    stack2 = stack[1:]

    for i, (slice1, slice2) in enumerate(zip(stack1, stack2)):
        zero_array = np.zeros((3, 512, 512))
        zero_array[0] = slice1
        zero_array[1] = slice2
        zero_array = np.ascontiguousarray(zero_array.transpose(1, 2, 0))
        consecutive_image = zero_array.astype('uint8')
        pictures = Image.fromarray(consecutive_image, 'RGB')
        pictures.save('%02d.png' % i)
        zero_array[0] = 0
        zero_array[1] = 0
        print(str(i + 1) + "/" + str(len(stack1)) + " files created")


def membrane_single_pictures():
    """
    - creating single membrane pictures for cytoplasma prediction
    """
    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)

    for i, slice in enumerate(stack):
        edges_compared = edges(slice)
        pictures = Image.fromarray(edges_compared.astype('uint8') * 255)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(stack)) + " files created")


def membrane_tif_stack():
    """
    - creating membrane tif stack for future usage in function membrane_consecutive
    - transpose function is needed for the correct 3D arrangement of the tif image stack
    """
    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)
    list_for_array = []

    for i, slice in enumerate(stack):
        edges_compared = edges(slice)
        edges_as_array = edges_compared.astype(int)
        list_for_array.append(edges_as_array)

    three_dim_array = np.dstack(list_for_array)
    three_dim_array = three_dim_array.transpose(2, 0, 1)
    membrane_to_tif = three_dim_array.astype('uint8') * 255
    tifsave('membrane-training-b.tif', membrane_to_tif)
    print('membrane-training-b.tif created')


def em_rgb():
    """
    - consecutive images are required for training with overlap images
    - picture i and picture i+1 are added to the red and green channel
    - transpose function is needed for the correct 3D arrangement of the .tif image stack
    """
    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)
    stack1 = stack[:-2]
    stack2 = stack[1:-1]
    stack3 = stack[2:]

    for i, (slice1, slice2, slice3) in enumerate(zip(stack1, stack2, stack3)):
        zero_array = np.zeros((3, 512, 512))
        zero_array[0] = slice1
        zero_array[1] = slice2
        zero_array[2] = slice3
        zero_array = np.ascontiguousarray(zero_array.transpose(1, 2, 0))
        consecutive_image = zero_array.astype('uint8')
        pictures = Image.fromarray(consecutive_image, 'RGB')
        pictures.save(str(i+1) + '.png')
        zero_array[0] = 0
        zero_array[1] = 0
        zero_array[2] = 0
        print(str(i + 1) + "/" + str(len(stack1)) + " files created")


def preprocessing_overlap_input():
    """
    function to create a tif stack from training1 output images (easy to process for "def consecutive membrane"),
    that are necessary to create consecutive images for the following training2
    NOTE: argument input_dir needs a "/" after directory name for glob.glob
    """
    image_names = []
    image_read = []
    directory = glob.glob(a.input_dir + '*.png')

    for file in directory:
        image_names.append(file)
    for image in image_names:
        image_read.append(imread(image))

    array = np.asarray(image_read)
    #try:
    #    array_dim_red = array[:, :, :, 0]
    #except:
    #    array_dim_red = array[:, :, :]

    os.chdir(a.output_dir)
    tifsave('cytoplasm_d_pred.tif', array)
    print('cytoplasm_predicted_for_overlap_c.tif created')


def cytoplasma():
    """
    - creating cytoplasma images for the cytoplasma prediction
    """
    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)

    for i, slice in enumerate(stack):
        edges_compared = edges(slice)
        cytoplasma_logical = np.logical_and(np.logical_not(edges_compared), slice > 0)
        pictures = Image.fromarray(cytoplasma_logical.astype('uint8') * 255)
        pictures.save('%02d.png' % i)
        print(str(i + 1) + "/" + str(len(stack)) + " files created")


def dilation():
    stack = tifread(a.input_dir)
    os.chdir(a.output_dir)
    liste = []

    for i, slice in enumerate(stack):
        edges_dilated = ndimage.generic_filter(slice, function=max, size=(3, 3))
        liste.append(edges_dilated)

    array = np.asarray(liste)
    tifsave('relabel_dilation_slice.tif', array)
    print('relabel_dilation_slice.tif created')


def main():

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.operation == 'split':
        prefix = a.split
        split_snemi3d(prefix)

    if a.operation == 'create_em':
        em()

    if a.operation == 'create_overlap':
        overlap()

    if a.operation == 'create_rgb':
        em_rgb()

    if a.operation == 'create_consecutive':
        consecutive()

    if a.operation == 'create_membrane':
        membrane_single_pictures()

    if a.operation == 'create_membrane_tif':
        membrane_tif_stack()

    if a.operation == 'create_cytoplasma':
        cytoplasma()

    if a.operation == 'preprocessing':
        preprocessing_overlap_input()

    if a.operation == 'preprocessing_membrane':
        membrane_single_pictures2()

    if a.operation == 'dilation':
        dilation()

    if a.operation == 'cutten':
        cutten()

    if a.operation == 'cutten2':
        cutten2()

    if a.operation == 'single':
        single()

    if a.operation == 'erosion':
        erosion()


main()
