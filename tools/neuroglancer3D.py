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
parser.add_argument("--em", required=False, help="path to em tif file")
parser.add_argument("--label", required=False, help="path to original labeled tif file")
parser.add_argument("--predicted", required=False, help="path to predicted tif file")

a = parser.parse_args()

if a.bind_address:
    neuroglancer.set_server_bind_address(a.bind_address)
if a.static_content_url:
    neuroglancer.set_static_content_source(url=a.static_content_url)


em_images = tifread(a.em)
em_array = np.asarray(em_images, dtype="uint8")

label_images = tifread(a.label)
label_array = np.asarray(label_images, dtype="uint16")

predicted_images = tifread(a.predicted)
predicted_array = np.asarray(predicted_images, dtype="uint16")

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    scaling = [5, 5, 30]
    offset = [0, 0, 0]

    s.layers["EM"] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(em_array, voxel_size=scaling, voxel_offset=offset),
    )
    s.layers["groundtruth"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(label_array, voxel_size=scaling, voxel_offset=offset),
    )
    s.layers["predicted"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(predicted_array, voxel_size=scaling, voxel_offset=offset),
    )

print(viewer)
