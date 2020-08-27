"""Copyright 2020 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Can be run as follows:
# python3 export_data.py --params_path=candid.json --export_folder=samples

import json
import argparse
import ee
from utils import TaskManager, SampleExporter, TropomiImagery, DSMImagery
from utils import RoadImagery, MultiSpectralImagery, WindImagery, stack_bands_from_imagery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export TFRecords from EE')
    parser.add_argument('--params_path', type=str,
                        help="json parameters file location")
    parser.add_argument('--start_date', type=str,
                        help="start of the date range to export in YYYY-MM-DD",
                        default="2018-01-01")
    parser.add_argument('--end_date', type=str,
                        help="end of the date range to export in YYYY-MM-DD",
                        default="2019-12-31")
    parser.add_argument('--params_path', type=str,
                        help="json parameters file location")
    parser.add_argument('--export_folder', type=str,
                        help="Cloud export folder")
    parser.add_argument('--bucket_name', type=str,
                        help="cloud bucket name",
                        default='aghriss-air-quality')
    parser.add_argument('--num_samples', type=int,
                        help="number of samples to export per image",
                        default=1000)
    parser.add_argument('--num_shards', type=int,
                        help="number of sharding of samples before export",
                        default=10)

    args = parser.parse_args()
    params_path = args.params_path
    start_date = args.start_date
    end_date = args.end_date
    bucket_name = args.bucket_name
    export_folder = args.export_folder
    num_samples = args.num_samples
    num_shards = args.num_shards

    # params formated as folllows:
    # params = {"collections" : collections_dict,
    #           "bands" : bands_dict,
    #           "kernel_radius" : int,
    #           "scale" : int}
    # both collections_dict and bands_dict have keys:
    # ["multispectral","tropomi", "dsm", "wind", "road"]
    # collection_dict values are collections' links
    # bands_dict values are lists of bands to select from each collectionn

    print("Loading", (params_path))
    params = json.load(open(params_path, 'r'))
    # patch_bands are the bands that will be exported as patches
    patch_bands = []
    patch_bands += params['bands']['multispectral']
    patch_bands += params['bands']['tropomi']
    patch_bands += params['bands']['road'] + params['bands']['dsm']
    patch_bands += ["%i_%s" % (i, b) for b in params['bands']['wind']
                    for i in range(12)]

    all_bands = patch_bands + ["HOD", "DOW", "DOM", "MOY", "latitude",
                               "longitude", "valid"]
    scale = params['scale']
    kernel_radius = params['kernel_radius']
    # Initialize EE
    ee.Authenticate()
    ee.Initialize()
    # Creating different kernels
    # square kernel used for neighboorhoodToArray patches computation
    neighborhood_kernel = ee.Kernel.square(kernel_radius)
    # vertical and horizontal kernels used in calculating valid pixels
    vertical_kernel = ee.Kernel.rectangle(xRadius=1,
                                          yRadius=kernel_radius,
                                          units='pixels')
    horizontal_kernel = ee.Kernel.rectangle(xRadius=kernel_radius,
                                            yRadius=1,
                                            units='pixels')
    # Initializing SampleExporter and TaskManager
    task_manager = TaskManager(verbose=True)
    sampler = SampleExporter(task_manager, num_samples, num_shards,
                             neighborhood_kernel, scale, bucket_name,
                             export_folder)
    # Initializing imagery classes
    multispectral = MultiSpectralImagery(params['collections']['multispectral'],
                                         start_date, end_date,
                                         bands=params['bands']['multispectral'],
                                         scale=scale)
    tropomi = TropomiImagery(params['collections']['tropomi'],
                             bands=params['bands']['tropomi'])
    # Wind imagery adds wind bands from the previous 12 hours
    wind = WindImagery(params['collections']['wind'],
                       bands=params['bands']['wind'])
    # DSM Imagery returns the most recent data prior to the tropomi date
    dsm = DSMImagery(params['collections']['dsm'], scale,
                     params['bands']['dsm'])
    # road is an image, returns clipped image cropped to multispectral region
    road = RoadImagery(params['collections']['road'],
                       bands=params['bands']['road'])

    print("Stacking bands for %i multispectral images" % len(multispectral))
    # We loop through Multispectral Images
    # First 2 images already exported
    for j in range(2, len(multispectral)):

        multispectral_image = multispectral[j]
        image_info = multispectral_image.getInfo()
        image_id = image_info['properties']['productionID']
        print("Exporting for Multispectral N:%i, %s" % (j, image_id))
        # ImageCollection of images that contain all_bands
        # ImageCollection is pre-filtered so that alll images have at least
        # num_samples valid pixels
        stacked_bands_images = stack_bands_from_imagery(multispectral_image,
                                                        wind, dsm, road,
                                                        tropomi,
                                                        vertical_kernel,
                                                        horizontal_kernel,
                                                        num_samples,
                                                        all_bands, scale)
        size = stacked_bands_images.size().getInfo()
        listed = stacked_bands_images.toList(size)
        print("%i has > %i valid points" % (size, num_samples))
        # We loop through the images with stacked bands
        for i in range(min(size, 2)):
            export_id = image_id + "_%i" % i
            sampler.export_patches(ee.Image(listed.get(i)),
                                   bands=all_bands,
                                   patch_bands=patch_bands,
                                   export_id=export_id)
