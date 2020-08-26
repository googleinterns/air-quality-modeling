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


class CollectionClass:
    """Base Imagery class for ImageCollection."""

    def __init__(self, link, bands=None):
        """Initialize the Collection class.

        Parameters
        ----------
        link : str
            ImageCollection link
        bands : list[str], optional
            Name of bands to select. The default is None to take all bands.

        """
        self.bands = bands
        self.imagery = ee.ImageCollection(link)
        if self.bands:
            self.imagery = self.imagery.select(self.bands)

    def get_clipped_bands(self, geometry):
        """Get clipped images after a filter bound on geometry.

        Parameters
        ----------
        geometry : ee.Geometry
            Geometry to clip to

        Returns
        -------
        ee.ImageCollection
            cliped images filterbounded on geometry

        """
        images = self.imagery.filterBounds(geometry)
        return images.map(lambda x: x.clip(geometry))

    def get_clipped_range_bands(self, geometry, start_date, end_date):
        """Get clipped images from a specific date range.

        Parameters
        ----------
        geometry : ee.Geometry
            DESCRIPTION.
        start_date : String/EE.Date
            DESCRIPTION.
        end_date : String/EE.Date
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        images = self.imagery.filterBounds(geometry)

        def clip_func(image):
            return image.clip(geometry)
        return images.filterDate(start_date, end_date).map(clip_func)


class RoadImagery:
    """Road Imagery Class: Wrapper for ee.Image."""

    def __init__(self, link, bands):
        """Initialize the Imagery Class.

        Parameters
        ----------
        link : str
            ee.Image link
        bands : list[str], optional
            Name of bands to select. The default is None to take all bands.

        Returns
        -------
        None.

        """
        self.bands = bands
        self.imagery = ee.Image(link).select(self.bands)

    def get_bands(self, geometry):
        """.

        Parameters
        ----------
        geometry : ee.Geometry
            Geometry to clip to


        Returns
        -------
        ee.Image
            Image clipped to the geometry

        """
        return self.imagery.clip(geometry).unmask(0, False)


class TropomiImagery(CollectionClass):
    """."""

    def __init__(self, link, bands=None, before_range=(0, 'day'),
                 after_range=(7, 'day')):
        """.

        Parameters
        ----------
        link : TYPE
            DESCRIPTION.
        bands : TYPE, optional
            DESCRIPTION. The default is None.
        before_range : TYPE, optional
            DESCRIPTION. The default is (0,'day').
        after_range : TYPE, optional
            DESCRIPTION. The default is (7,'day').

        Returns
        -------
        None.

        """
        super(TropomiImagery, self).__init__(link, bands)
        self.before = before_range
        self.after = after_range

    def get_bands(self, date, geometry):
        """."""
        return self.get_clipped_range_bands(geometry,
                                            date.advance(*self.before),
                                            date.advance(*self.after))


class MultiSpectralImagery(CollectionClass):
    """."""

    def __init__(self, link, start_date, end_date, bands, scale):
        """.

        Parameters
        ----------
        link : TYPE
            DESCRIPTION.
        start_date : TYPE
            DESCRIPTION.
        end_date : TYPE
            DESCRIPTION.
        bands : TYPE
            DESCRIPTION.
        scale : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(MultiSpectralImagery, self).__init__(link, bands)

        self.imagery = self.imagery.filterDate(start_date, end_date)
        self.imagery = self.imagery.filter(ee.Filter.notNull(
            ['collectionStartTime']))
        self.size = self.imagery.size().getInfo()
        self.imagery_list = self.imagery.toList(self.size)
        print("Base Imagery has %i elements" % self.size)
        self.scale = scale

    def __getitem__(self, i):
        """.

        Parameters
        ----------
        i : int
            index of element to return

        Returns
        -------
        ee.Image

        """
        assert i < len(self)
        image = ee.Image(self.imagery_list.get(i))
        return image.clipToBoundsAndScale(image.geometry(), scale=self.scale)

    def __len__(self):
        """Return length of collection list."""
        return self.size


class DSMImagery(CollectionClass):
    """DSM Imagery wrapper."""

    def get_bands(self, date, geometry, scale):
        """DSM is always scaled, hence the need for scale argument.

        Parameters
        ----------
        date : ee.Date
            Date to which most recent imagery is taken
        geometry : ee.Geometry
            Geometry to which images are clipped
        scale : int, optional
            Scale of the images (m/pixel)

        Returns
        -------
        ee.Image
            Image with most recent data before the given date

        """
        images = self.imagery.filterBounds(geometry)

        def add_hour_difference(img):
            return img.setMulti({"hour_difference": date.difference(img.date(),
                                                                    'hour')})

        images = images.map(add_hour_difference)
        images = images.filter(ee.Filter.gt('hour_difference', 0))
        clip_scale_func = clip_and_scale(geometry, scale)
        images = images.sort('hour_difference', True).map(clip_scale_func)
        image = images.reduce(ee.Reducer.firstNonNull())
        image = image.select(["%s_first" % b for b in self.bands], self.bands)
        return image


class WindImagery(CollectionClass):
    """."""

    def __init__(self, link, bands=None, before_range=(-12, 'hour')):
        """.

        Parameters
        ----------
        link : str
        bands : list[str] optional
        before_range : tuple, optional
            Extent of wind bands to take before givne date.
            The default is (-12, 'hour').
        """
        super(WindImagery, self).__init__(link, bands)
        self.before = before_range

    def get_bands(self, date, geometry):
        """Get stacked bands from the wind imagery range.

        Parameters
        ----------
        date : ee.Date
        geometry : ee.Geometry

        Returns
        -------
        ee.Image
            Wind bands where i_b is value of band "b" "i" hours before date.

        """
        images = self.get_clipped_bands(geometry)
        images = images.filter(ee.Filter.date(date.advance(*self.before), date)
                               )
        images = images.map(lambda img: self._stack_bands(img, date).clip(
            geometry))

        return merge_bands(images)

    def _stack_bands(self, img, date):
        """Stack bands from different images, helper method.

        Parameters
        ----------
        img : ee.Image
        date : ee.Date

        Returns
        -------
        ee.Image
            Image with renamed band preficed by time difference in hours

        """
        return img.select(self.bands,
                          [ee.Algorithms.String(date.difference(img.date(),
                                                                'hour').int()
                                                ).cat('_%s' % b)
                           for b in self.bands])


def merge_bands(images):
    """Create image from collection by stacking the bands from the images.

    Parameters
    ----------
    images : ee.ImageCollection
        The image collection from which bands will be stacked

    Returns
    -------
    ee.Image
        Image with stacked bands

    """
    def stack_bands(new, previous):
        return ee.Image(previous).addBands(new)

    return ee.Image(images.iterate(stack_bands, ee.Image(None)))


def clip_and_scale(geometry, scale):
    """Return clipToBoundsAndScale map function.

    Parameters
    ----------
    geometry : ee.Geometry
    scale : int


    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    def clip_scale(image):
        return image.clipToBoundsAndScale(geometry, scale=scale)
    return clip_scale
