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

import json
import argparse
import ee
from utils import TaskManager, SampleExporter, images_with_stacked_bands
from utils import RoadImagery, MultiSpectralImagery, TropomiImagery, DSMImagery, WindImagery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export Data from EarthEngine'
                                     )
    parser.add_argument('--params_path', type=str,
                        help="json parameters file location")
    parser.add_argument('--export_folder', type=str,
                        help="Cloud export folder")
    parser.add_argument('--bucket', type=str,
                        help="bucket name",
                        default='aghriss-air-quality')
    parser.add_argument('--num_samples', type=int,
                        help="number of samples to export per image",
                        default=1000)
    parser.add_argument('--num_shards', type=int,
                        help="number of sharding of samples before export",
                        default=10)

    args = parser.parse_args()

    params = json.load(open(args.params_path, 'r'))
    BUCKET = args.bucket
    DIRECTORY = args.export_folder
    num_samples = args.num_samples
    num_shards = args.num_shards

    print("Loading", (args.params_path))
    params_path = args.params_path
    params = json.load(open(params_path, 'r'))
    PATCH_BANDS = []
    PATCH_BANDS += params['bands']['multispectral']
    PATCH_BANDS += params['bands']['tropomi']
    PATCH_BANDS += params['bands']['road'] + params['bands']['dsm']
    STACKED_WIND_BANDS = ["%i_%s" % (i, b) for b in params['bands']['wind']
                          for i in range(12)]
    PATCH_BANDS += STACKED_WIND_BANDS
    BANDS = PATCH_BANDS + ["HOD", "DOW", "DOM", "MOY", "latitude", "longitude",
                           "valid"]
    SCALE = params['scale']
    KERNEL_RADIUS = params['kernel_radius']
    # Initialize EE
    ee.Authenticate()
    ee.Initialize()
    # Creating diffrent kernels
    kernel = ee.Kernel.square(KERNEL_RADIUS)
    # vertical and horizontal kernel used in calculating valid pixels
    vertical_kernel = ee.Kernel.rectangle(xRadius=1,
                                          yRadius=KERNEL_RADIUS,
                                          units='pixels')
    horizontal_kernel = ee.Kernel.rectangle(xRadius=KERNEL_RADIUS,
                                            yRadius=1,
                                            units='pixels')
    # Initializing smaple exporter and TaskManager
    task_manager = TaskManager(verbose=True)
    sampler = SampleExporter(task_manager, num_samples, num_shards,
                             kernel, SCALE, BUCKET, DIRECTORY)
    # Initializing imagery classes
    multispectral = MultiSpectralImagery(params['collections']['multispectral'],
                                         start_date="2018-01-01",
                                         end_date="2019-12-31",
                                         bands=params['bands']['multispectral'],
                                         scale=SCALE)
    tropomi = TropomiImagery(params['collections']['tropomi'],
                             bands=params['bands']['tropomi'])
    # Wind imagery adds wind bands from the previous 12 hours
    wind = WindImagery(params['collections']['wind'],
                       bands=params['bands']['wind'])
    # DSM Imagery returns the most recent data prior to the tropomi date
    dsm = DSMImagery(params['collections']['dsm'],
                     bands=params['bands']['dsm'])
    # road is an image, returns clipped image cropped to multispectral region
    road = RoadImagery(params['collections']['road'],
                       bands=params['bands']['road'])

    print("Stacking bands for %i multispectral images" % len(multispectral))

    # First 2 images already exported
    for j in range(2, len(multispectral)):

        multispectral_image = multispectral[j]
        image_info = multispectral_image.getInfo()
        image_id = image_info['properties']['productionID']

        print("Exporting for Multispectral N:%i, %s" % (j, image_id))

        stacked_bands_images = images_with_stacked_bands(multispectral_image,
                                                         wind, dsm, road,
                                                         tropomi,
                                                         vertical_kernel,
                                                         horizontal_kernel,
                                                         num_samples,
                                                         BANDS, SCALE)
        size = stacked_bands_images.size().getInfo()
        listed = stacked_bands_images.toList(size)
        print("%i has > %i valid points" % (size, num_samples))
        for i in range(min(size, 2)):
            export_id = image_id+"_"+str(i)
            sampler.export_patches(ee.Image(listed.get(i)),
                                   bands=BANDS,
                                   patch_bands=PATCH_BANDS,
                                   export_id=export_id)

