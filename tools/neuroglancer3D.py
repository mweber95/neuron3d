from __future__ import print_function
from tifffile import imread as tifread
import argparse
import numpy as np
import neuroglancer

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--bind-address",
    help="Bind address for Python web server.  Use 127.0.0.1 (the default) to restrict access "
    "to browers running on the local machine, use 0.0.0.0 to permit access from remote browsers.")
parser.add_argument(
    "--static-content-url", help="Obtain the Neuroglancer client code from the specified URL.")
parser.add_argument("--input_em", required=False, help="path to em.tif")
parser.add_argument("--input_label", required=True, help="path to label.tif")
parser.add_argument("--input_relabeled", required=False, help="path to relabeled.tif")
parser.add_argument("--input_relabeled_2", required=False, help="path to relabeled.tif")
parser.add_argument("--input_relabeled_3", required=True, help="path to relabeled.tif")

a = parser.parse_args()

if a.bind_address:
    neuroglancer.set_server_bind_address(a.bind_address)
if a.static_content_url:
    neuroglancer.set_static_content_source(url=a.static_content_url)


em_images = tifread(a.input_em)
em_images = np.asarray(em_images, dtype="uint8")

ground_truth = tifread(a.input_label)
ground_truth = np.asarray(ground_truth, dtype="uint16")

#relabeled = tifread(a.input_relabeled)
#relabeled = np.asarray(relabeled, dtype="uint16")

#relabeled_stack = tifread(a.input_relabeled_2)
#relabeled_stack = np.asarray(relabeled_stack, dtype="uint16")

relabeled_slice = tifread(a.input_relabeled_3)
relabeled_slice = np.asarray(relabeled_slice, dtype="uint16")

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    scaling = [5, 5, 30]
    offset = [0, 0, 0]

    s.layers["EM"] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(em_images, voxel_size=scaling, voxel_offset=offset),
    )
    s.layers["ground_truth"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(ground_truth, voxel_size=scaling, voxel_offset=offset),
    )
    # s.layers["relabeled"] = neuroglancer.SegmentationLayer(
    #     source=neuroglancer.LocalVolume(relabeled, voxel_size=scaling, voxel_offset=offset),
    # )
    # s.layers["relabeled_stack_dilated"] = neuroglancer.SegmentationLayer(
    #     source=neuroglancer.LocalVolume(relabeled_stack, voxel_size=scaling, voxel_offset=offset),
    # )
    s.layers["relabeled"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(relabeled_slice, voxel_size=scaling, voxel_offset=offset),
    )
    # s.voxel_coordinates = offset

print(viewer)
