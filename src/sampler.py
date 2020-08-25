import ee
import time
class SampleExporter(object):

    def __init__(self, num_shards, task_manager, bucket, directory):
        self.num_shards = num_shards
        #self.features =  features
        self.task_manager = task_manager
        self.bucket = bucket
        self.directory = directory

    
    
    def export_patches(self, image, bands, patch_bands, n_samples, kernel, scale,
                       export_id):
        valid_pixels = image.select('valid').int()
        valid_pixels = valid_pixels.updateMask(valid_pixels)
        valid_samples = valid_pixels.stratifiedSample(numPoints=n_samples,
                                                      classBand="valid",
                                                      region=image.geometry(),
                                                      scale=scale,
                                                      tileScale=2,
                                                      geometries=True)

        patch_names = ["patch_%s" % b for b in patch_bands]
        patches = image.select(patch_bands).neighborhoodToArray(kernel).select(
            patch_bands,patch_names)
        
        combined = ee.Image.cat([image,patches])
        def get_sample(feature):
            return combined.sample(numPixels=1,
                                   region=feature.geometry(),
                                   scale=scale).first()
        samples = valid_samples.map(get_sample)

        self.export_tasks(samples, bands+patch_names, export_id)
        
    def export_tasks(self, samples, features, export_id):
        samples_for_sharding = samples.randomColumn('shard_split')
        for i in range(self.num_shards):
            range_min = float(i)/float(self.num_shards)
            range_max = float(i+1)/float(self.num_shards)
            range_filter = ee.Filter.And(ee.Filter.gte('shard_split', range_min), ee.Filter.lt('shard_split', range_max))
            samples_to_export = samples_for_sharding.filter(range_filter)
        

            task = ee.batch.Export.table.toCloudStorage(
                 collection = samples_to_export,
                 description = export_id+"_%i"%i,
                  bucket = self.bucket,
                  fileNamePrefix = self.directory + '/' + export_id+"_%i"%i,
                  fileFormat = 'TFRecord',
                  selectors = features,
                  maxWorkers=2000
              )
            while self.task_manager.busy(): time.sleep(1)
            self.task_manager.submit(task)

