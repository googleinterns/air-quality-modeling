# Import, authenticate and initialize the Earth Engine library.

import ee

def create_images(params):
    kernel = ee.Kernel.square(params['kernel_size'])
    
    collections = {k:ee.ImageCollection(
        params['collections'][k]).select(params['bands'][k]) 
                                        for k in params['collections'].keys()}
    for k in params['rename'].keys():
        collections[k] = collections[k].select(params['bands'][k],
                                                       params['rename'][k])
    collections['ortho'] = collections['ortho'].filter(
                                    ee.Filter.notNull(['collectionEndTime']))

    def create_patches(img):
        mask = img.reduce(ee.Reducer.anyNonZero());
        end_time = ee.Date(img.get('collectionEndTime'));
        count, period= params['range']['tropomi']
        tropomi_img = collections['tropomi'].filterDate(
            end_time.advance(count,period), end_time).mean()
          
        ortho_img = img.neighborhoodToArray(kernel)
        ortho_img = ortho_img.select(params['patch'],['patch_'+ b
                                                for b in params['patch']]);
        ortho_tropo = ee.Image.cat([ortho_img, tropomi_img])

        res = ee.Algorithms.If(tropomi_img.bandNames().length().neq
                               (ee.Number(0)),ortho_tropo.updateMask(mask), None);
        return res

    ortho_tropomi_coll = collections['ortho'].map(create_patches,True)
    ortho_tropomi_list = ortho_tropomi_coll.toList(ortho_tropomi_coll.size())
    return ortho_tropomi_list

def create_tasks(images, params, task_manager):

    SCALE = params['scale']
    for i in range(0,648):
      img = ee.Image(candid_patches_list.get(i))
      geoSample = ee.FeatureCollection([])
      for s in range(SHARD):
        sample = img.sample(region=img.geometry(),
                            numPixels=SAMPLES_PER_IMG//SHARD,
                            scale=SCALE,
                            seed=s)  
        geoSample = geoSample.merge(sample)
    
      task = ee.batch.Export.table.toCloudStorage(
        collection = geoSample,
        description = DESC%i,
        bucket = BUCKET,
        fileNamePrefix = FOLDER + '/' + DESC%i,
        fileFormat = 'TFRecord')
      task.start()
      
      
    ee.Authenticate()
    ee.Initialize()
