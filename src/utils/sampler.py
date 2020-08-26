"""
Copyright 2020 Google LLC.

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

import ee


class SampleExporter:
    """Manage export operations, takes images to extract patches."""

    def __init__(self, task_manager, num_samples, num_shards, kernel, scale,
                 bucket, directory):
        """Initialize the SampleExport object.

        Parameters
        ----------
        task_manager : TaskManager
            TaskManager object to submit the tasks to
        num_samples : int
            Number of samples to extract from each image.
        num_shards : int
            Number of shards used to parition the samples.
        kernel : ee.Kernel
            kernel used for neighborhoodtoArray
        scale : int
            scale parameter for sampling and export (in meters/pixel)
        bucket : str
            Google Cloud Storage Bucke name
        directory : str
            Folder to extract the TFRecords to

        Returns
        -------
        None.

        """
        # Task parameters
        self.task_manager = task_manager
        self.num_samples = num_samples
        self.num_shards = num_shards

        # Patches paramters
        self.kernel = kernel
        self.scale = scale

        # Storage location
        self.bucket = bucket
        self.directory = directory

    def export_patches(self, image, bands, patch_bands, export_id):
        """Export patch and scalar bands and submit tasks to the task manager.

        Parameters
        ----------
        image : ee.Image
            Image from which to extract patches. It should have a 'valid' band.
        bands : list[str]
            bands exported as scalars
        patch_bands : list[str]
            bands exported as patches
        export_id : str
            prefix for the task description

        Returns
        -------
        None

        """
        valid_pixels = image.select('valid').int()
        valid_pixels = valid_pixels.updateMask(valid_pixels)
        # Extract points from the valid pixels
        valid_samples = valid_pixels.stratifiedSample(
            numPoints=self.num_samples,
            classBand="valid",
            region=image.geometry(),
            scale=self.scale,
            tileScale=2,
            geometries=True)

        patch_names = ["patch_%s" % b for b in patch_bands]
        patches = image.select(patch_bands).neighborhoodToArray(self.kernel)
        patches = patches.select(patch_bands, patch_names)

        combined = image.addBands(patches)

        def get_sample(feature):
            return combined.sample(numPixels=1,
                                   region=feature.geometry(),
                                   scale=self.scale).first()

        samples = valid_samples.map(get_sample)
        features = bands + patch_names
        self.export_tasks(samples, features, export_id)

    def export_tasks(self, samples, features, export_id):
        """Shard the samples into num_shards and submit tasks to TaskManager.

        Parameters
        ----------
        samples : ee.FeatureCollection
            Collection of samples to export
        features : list[str]
            features to export from samples
        export_id : str
            prefix for the task description

        Returns
        -------
        None.

        """
        samples_for_sharding = samples.randomColumn('shard_split')
        for i in range(self.num_shards):
            range_min = float(i)/float(self.num_shards)
            range_max = float(i+1)/float(self.num_shards)
            range_filter = ee.Filter.And(
                ee.Filter.gte('shard_split', range_min),
                ee.Filter.lt('shard_split', range_max))
            samples_to_export = samples_for_sharding.filter(range_filter)

            task = ee.batch.Export.table.toCloudStorage(
                collection=samples_to_export,
                description=export_id+"_%i" % i,
                bucket=self.bucket,
                fileNamePrefix=self.directory + '/' + export_id+"_%i" % i,
                fileFormat='TFRecord',
                selectors=features,
                maxWorkers=2000)

            # Can be a stopping call if TaskManager if busy.
            self.task_manager.submit(task)
