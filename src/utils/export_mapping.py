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


# Function uses Earth Engine API
# Contains more local variables than rocommended in a Python code
def images_with_stacked_bands(multispectral_image, wind, dsm, road, tropomi,
                              vertical_kernel, horizontal_kernel, num_samples,
                              bands, scale):
    """Stack bands from different imageries.

    Parameters
    ----------
    multispectral_image : ee.Image
        Multispectral Image
    wind : CollectionClass
        collectionClass for Wind Imagery
    dsm : CollectionClass
        collection class for DSM Imagery
    road : RoadImagery
        Road Imagery
    tropomi : CollectionClass
        Tropomi imagery
    vertical_kernel : ee.Kernel
        kernel used for convolved mask to calculate valid pixels
    horizontal_kernel : ee.Kernel
        horizontal kernel for convolution
    num_samples : int
        Minimum number of valid pixels that stacked image shoudl have
    bands : list[str]
        list of bands that the image shoudl contain
    scale : int
        Scale for clipping and sampling

    Returns
    -------
    ee.ImageCollection
        Collection of images that contain all requested bands with num_samples
        valid pixels.

    """
    start_time = multispectral_image.get('collectionStartTime')
    multispectral_date = ee.Date(start_time)
    multispectral_geometry = multispectral_image.geometry()
    multispectral_mask = multispectral_image.mask()
    multispectral_mask = multispectral_mask.reduce(ee.Reducer.anyNonZero())

    tropomi_images = tropomi.get_bands(multispectral_date,
                                       multispectral_geometry)
    road_bands = road.get_bands(multispectral_geometry)
    latlon = ee.Image.pixelLonLat().updateMask(multispectral_mask)
    latlon = latlon.clipToBoundsAndScale(multispectral_geometry,
                                         scale=scale)

    def add_bands(tropomi_image):
        """Match bands from different imageries."""
        tropomi_date = tropomi_image.date()
        tropomi_mask = tropomi_image.mask().reduce(ee.Reducer.anyNonZero())

        wind_bands = wind.get_bands(tropomi_date, multispectral_geometry)
        dsm_bands = dsm.get_bands(tropomi_date,
                                  multispectral_geometry,
                                  scale=scale)

        total_mask = multispectral_mask.addBands(tropomi_mask)
        total_mask = total_mask.reduce(ee.Reducer.allNonZero())

        date_bands = add_day_bands(tropomi_date, total_mask)
        date_bands = date_bands.clipToBoundsAndScale(multispectral_geometry,
                                                     scale=scale)

        convolved_mask = total_mask.convolve(vertical_kernel)
        convolved_mask = convolved_mask.convolve(horizontal_kernel)
        valid_neighborhood = convolved_mask.eq(1).unmask(0, False)
        valid_neighborhood = valid_neighborhood.rename(['valid'])

        valid_pixels = valid_neighborhood.reduceRegion(ee.Reducer.sum(),
                                                       multispectral_geometry,
                                                       scale)
        combined_bands = ee.Image.cat([multispectral_image, tropomi_image,
                                       dsm_bands, wind_bands, road_bands,
                                       date_bands, latlon,
                                       valid_neighborhood])
        combined_bands = combined_bands.updateMask(total_mask)
        combined_bands = combined_bands.select(bands)

        num_valid_pixels = ee.Number(valid_pixels.get('valid'))

        return ee.Algorithms.If(num_valid_pixels.gt(num_samples),
                                combined_bands,
                                None)
    valid_stacked_bands = tropomi_images.map(add_bands, opt_dropNulls=True)
    return valid_stacked_bands


def date_band(date, unit, interval, name):
    """Return an Image with relative value of unit in interval."""
    return ee.Image.constant(date.getRelative(unit, interval)).rename(name)


def add_day_bands(date, mask):
    """Stack relative datevalue of different intervals."""
    hour_of_day = date_band(date, 'hour', 'day', 'HOD').updateMask(mask)
    day_of_week = date_band(date, 'day', 'week', 'DOW').updateMask(mask)
    day_of_month = date_band(date, 'day', 'week', 'DOM').updateMask(mask)
    month_of_year = date_band(date, 'month', 'year', 'MOY').updateMask(mask
                                                                       )
    return ee.Image.cat([hour_of_day, day_of_week,
                         day_of_month, month_of_year])
