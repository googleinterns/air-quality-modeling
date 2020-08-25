'''
Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import ee
import json
import argparse
import wrappers
from taskmanager import TaskManager
from sampler import SampleExporter





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export Data from EarthEngine'
                                     )
    parser.add_argument('--params_path', type=str)
    parser.add_argument('--export_folder', type=str)
    parser.add_argument('--bucket', type=str, default='aghriss-air-quality')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--shards', type=int, default=10)

    args = parser.parse_args()


    params = json.load(open(args.params_path, 'r'))
    bucket = args.bucket
    export_folder = args.export_folder
    n_samples = args.samples
    n_shards = args.shards

    print("Loading", (args.params_path))
    params_path = args.params_path
    params = json.load(open(params_path, 'r'))
    PATCH_BANDS = params['bands']['multispectral'] + \
        params['bands']['tropomi'] + \
        params['bands']['road'] + params['bands']['dsm'] + \
        ["%i_%s" % (i, b) for b in params['bands']['wind'] for i in range(12)]

    BANDS = PATCH_BANDS + ["HOD", "DOW", "DOM", "MOY", "latitude", "longitude", "valid"]
 
    SCALE = params['scale']
    KERNEL_RADIUS = params['kernel_radius']

    ee.Authenticate()
    ee.Initialize()


    task_manager = TaskManager(verbose=True)
    sampler = SampleExporter(num_shards=n_shards,task_manager=task_manager,
			     bucket=bucket,
                             directory=export_folder)

    multispectral = wrappers.MultiSpectralImagery(
        link=params['collections']['multispectral'],
        start_date="2018-01-01",
        end_date="2019-12-31",
        bands=params['bands']['multispectral'])
    tropomi = wrappers.TropomiImagery(link=params['collections']['tropomi'],
                                      bands=params['bands']['tropomi'])
    wind = wrappers.WindImagery(link=params['collections']['wind'],
                                bands=params['bands']['wind'])
    dsm = wrappers.DSMImagery(link=params['collections']['dsm'],
                              bands=params['bands']['dsm'])
    road = wrappers.RoadImagery(link=params['collections']['road'],
                                bands=params['bands']['road'])

    kernel = ee.Kernel.square(KERNEL_RADIUS)

    def date_band(date, unit, interval, name):
        return ee.Image.constant(date.getRelative(unit, interval)).rename(name)

    def add_day_bands(date, mask):
        hour_of_day = date_band(date, 'hour', 'day', 'HOD').updateMask(mask)
        day_of_week = date_band(date, 'day', 'week', 'DOW').updateMask(mask)
        day_of_month = date_band(date, 'day', 'week', 'DOM').updateMask(mask)
        month_of_year = date_band(date, 'month', 'year', 'MOY').updateMask(mask
                                                                           )
        return ee.Image.cat([hour_of_day, day_of_week,
                             day_of_month, month_of_year])
    print("Exporting %i multispectral"% len(multispectral))
    for j in range(2, len(multispectral)):

        spectral_image = multispectral[j]
        date = ee.Date(spectral_image.get('collectionStartTime'))
        geometry = spectral_image.geometry()
        mask = spectral_image.mask().reduce(ee.Reducer.anyNonZero())
        imageInfo = spectral_image.getInfo()
        export_id = imageInfo['properties']['productionID']

        print("Exporting for Multispectral %s"%export_id)

        multi_bands = spectral_image.clipToBoundsAndScale(geometry, scale=SCALE)
        multi_mask = mask.clipToBoundsAndScale(geometry, scale=SCALE)
        tropo = tropomi.get_bands(date, geometry)
        tropomi_image = tropo.first()
        road_bands = road.get_bands(geometry)
        latlon = ee.Image.pixelLonLat().updateMask(mask).clipToBoundsAndScale(
            geometry, scale=SCALE)

        def add_bands(tropomi_image):
            tropomi_bands = tropomi_image.clip(geometry)
            tropomi_date = tropomi_bands.date()
            tropomi_mask = tropomi_bands.mask().reduce(ee.Reducer.anyNonZero())
            wind_bands = wind.get_bands(tropomi_date, geometry)
            dsm_bands = dsm.get_bands(tropomi_date, geometry).clipToBoundsAndScale(geometry, scale=SCALE)
            total_mask = multi_mask.float().addBands(tropomi_mask).reduce(
                                                            ee.Reducer.allNonZero()
                                                            )
            cropped_mask = total_mask.clipToBoundsAndScale(geometry, scale=SCALE)
            convolved_mask = cropped_mask.convolve(
                ee.Kernel.rectangle(xRadius=1,
                                    yRadius=KERNEL_RADIUS,
                                    units='pixels'))
            convolved_mask = convolved_mask.convolve(
                ee.Kernel.rectangle(xRadius=KERNEL_RADIUS,
                                    yRadius=1,
                                    units='pixels'))
            valid_neighborhood = convolved_mask.eq(1).unmask(0, False).rename(
                ['valid'])

            n_valid = valid_neighborhood.reduceRegion(ee.Reducer.sum(),
                                                      geometry,
                                                      SCALE)
            combined = ee.Image.cat([multi_bands, tropomi_bands,
                                              dsm_bands, wind_bands,
                                              road_bands,
                                              add_day_bands(tropomi_date,
                                                            total_mask).clipToBoundsAndScale(
							geometry, scale=SCALE),
                                              latlon,
                                              valid_neighborhood]).updateMask(
                                                      total_mask).select(BANDS)
            return ee.Algorithms.If(ee.Number(n_valid.get('valid')).gt(n_samples),
                                    combined,
                                    None)

        tropo_filtered = tropo.map(add_bands, opt_dropNulls=True)
        size = tropo_filtered.size().getInfo()
        listed = tropo_filtered.toList(size)
        print("%i has > %i valid points" % (size, n_samples))
        for i in range(min(size,2)):
            sampler.export_patches(ee.Image(listed.get(i)),
                                   bands=BANDS,
                                   patch_bands=PATCH_BANDS,
                                   n_samples=n_samples,
                                   kernel=kernel,
                                   scale=SCALE,
                                   export_id=export_id+"_"+str(i))
