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
ee.Authenticate()
ee.Initialize()
import utils
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export Data from EarthEngine')
    parser.add_argument('--params_path', type=str)
    parser.add_argument('--export_folder', type=str)
    parser.add_argument('--bucket', type=str)    
    parser.add_argument('--samples', type=int)
    parser.add_argument('--shards', type=int)
    
    args = parser.parse_args()
    print("Loading",(args.params_path))
    params = json.load(open(args.params_path,'r'))
    bucket = args.bucket
    export_folder = args.export_folder
    n_samples = args.samples
    n_shards = args.shards
        
    params = json.load(open('export_params/params.json', 'r'))
    PATCH_BANDS = params['bands']['multispectral'] + params['bands']['tropomi'] + \
        params['bands']['road'] + params['bands']['dsm'] + \
        ["%i_%s"%(i, b) for b in params['bands']['wind'] for i in range(12)]
    
    BANDS = PATCH_BANDS + ["HOD", "DOW", "DOM", "MOY", 'valid']
    SCALE = params['scale']
    KERNEL_RADIUS = params['kernel_radius']
    
    multispectral = utils.MultiSpectralImagery(link = params['collections']['multispectral'],
                                               start_date = "2018-01-01",
                                               end_date = "2019-12-31",
                                               bands =params['bands']['multispectral'])
    tropomi = utils.TropomiImagery(link = params['collections']['tropomi'],
                                   bands = params['bands']['tropomi'])
    wind = utils.WindImagery(link = params['collections']['wind'],
                             bands = params['bands']['wind'])
    dsm = utils.DSMImagery(link = params['collections']['dsm'],
                             bands = params['bands']['dsm'])
    road = utils.RoadImagery(link = params['collections']['road'],
                             bands = params['bands']['road'])
    
    
    kernel = ee.Kernel.square(KERNEL_RADIUS)
    
    def date_band(date,unit,interval,name):
        return ee.Image.constant(date.getRelative(unit, interval)).rename(name)
    
    def add_day_bands(date, mask):
        hour_of_day = date_band(date, 'hour', 'day', 'HOD').updateMask(mask)
        day_of_week = date_band(date, 'day', 'week', 'DOW').updateMask(mask)
        day_of_month = date_band(date, 'day', 'week', 'DOM').updateMask(mask)
        month_of_year = date_band(date, 'month', 'year', 'MOY').updateMask(mask)
        return ee.Image.cat([hour_of_day, day_of_week,
                                             day_of_month, month_of_year])
    
    spectral_image = multispectral[0]
    date = ee.Date(spectral_image.get('collectionStartTime'))
    geometry = spectral_image.geometry()
    mask = spectral_image.mask().reduce(ee.Reducer.anyNonZero())
    imageInfo = spectral_image.getInfo()
    export_id = imageInfo['properties']['productionID']
    multi_bands = spectral_image.clipToBoundsAndScale(geometry, scale=SCALE)
    multi_mask = mask.clipToBoundsAndScale(geometry, scale=SCALE)
    tropo = tropomi.get_bands(date, geometry)
    tropomi_image = tropo.first()
    road_bands = road.get_bands(geometry)
    latlon = ee.Image.pixelLonLat().updateMask(mask
                                ).clipToBoundsAndScale(geometry, scale=SCALE)
    
    def add_bands(tropomi_image):
        
        
        tropomi_bands = tropomi_image.clip(geometry)  
        tropomi_date = tropomi_bands.date()
        tropomi_mask = tropomi_bands.mask().reduce(ee.Reducer.anyNonZero())
        wind_bands = wind.get_bands(tropomi_date, geometry)
        dsm_bands = dsm.get_bands(tropomi_date, geometry)
        total_mask = multi_mask.float().addBands(tropomi_mask).reduce(
                                                        ee.Reducer.allNonZero())
        cropped_mask = total_mask.clipToBoundsAndScale(geometry, scale=SCALE)
        convolved_mask = cropped_mask.convolve(ee.Kernel.rectangle(xRadius=1, yRadius=KERNEL_RADIUS, units = 'pixels'));
        convolved_mask = convolved_mask.convolve(ee.Kernel.rectangle(xRadius=KERNEL_RADIUS, yRadius= 1, units ='pixels'))
        valid_neighborhood = convolved_mask.eq(1).unmask(0, False).rename(['valid'])
        
        n_valid = valid_neighborhood.reduceRegion(ee.Reducer.sum(),geometry,SCALE)
        
        combined = ee.Image(ee.Image.cat([multi_bands, tropomi_bands,
                                 dsm_bands, wind_bands, road_bands,
                                 add_day_bands(tropomi_date,total_mask),
                                 latlon,
                                 valid_neighborhood]).setMulti(n_valid)).mask(total_mask).select(BANDS)
        return ee.Algorithms.If(ee.Number(combined.get('valid')).gt(1000),combined, None)
    
    tropo_filtered = tropo.map(add_bands,opt_dropNulls=True)
    size = tropo_filtered.size().getInfo()
    listed = tropo_filtered.toList(size)
    
    from utils.task_queue import TaskManager
    from utils.sampler import SampleExporter
    
    task_manager = TaskManager()
    sampler = SampleExporter(num_shards = n_shards, features = BANDS+PATCH_BANDS,
                             task_manager=task_manager, bucket=bucket,
                             directory=export_folder)
    sampler.export_patches(ee.Image(listed.get(0)),
                           bands=BANDS,
                           patch_bands=PATCH_BANDS,
                           n_samples=1000,
                           kernel=kernel,
                           scale=SCALE,
                           export_id = export_id+"_"+str(0))
    
    task_manager.waiting_tasks[0].start()