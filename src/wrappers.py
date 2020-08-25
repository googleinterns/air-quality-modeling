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


class CollectionClass(object):
    """Base Imagery class for ImageCollection."""

    def __init__(self, link, bands=None):
        """Initialize the Collection class.

        Parameters
        ----------
        link : TYPE
            DESCRIPTION.
        bands : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.bands = bands
        self.imagery = ee.ImageCollection(link)
        if self.bands:
            self.imagery = self.imagery.select(self.bands)

    def get_cropped_bands(self, geometry, scale):
        """Clip and Scale the images after bound filter on geometry.

        Parameters
        ----------
        geometry : ee.Geometry
            Geometry of the bounds.
        scale : int
            the value of the scale to use in clipToBoundsAndScale
        Returns
        -------
        None

        """
        return self.imagery.filterBounds(geometry).clipToBoundsAndScale(
            geometry, scale=scale)

    def get_clipped_bands(self, geometry):
        """Get clipped images after a filter bound on geometry.

        Parameters
        ----------
        geometry : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.imagery.filterBounds(geometry).map(
            lambda x: x.clip(geometry))

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
        return self.imagery.filterBounds(geometry).filterDate(start_date,
                                                              end_date).map(
                                                  lambda x: x.clip(geometry))


class RoadImagery(object):
    """Road Imagery Class: Wrapper for ee.Image."""

    def __init__(self, link, bands):
        """Initialize the Imagery Class.

        Parameters
        ----------
        link : TYPE
            DESCRIPTION.
        bands : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.bands = bands
        self.imagery = ee.Image(link).select(bands)

    def get_bands(self, geometry):
        """.

        Parameters
        ----------
        geometry : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

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

    def __init__(self, link, start_date, end_date, bands=None):
        """.

        Parameters
        ----------
        link : TYPE
            DESCRIPTION.
        start_date : TYPE
            DESCRIPTION.
        end_date : TYPE
            DESCRIPTION.
        bands : TYPE, optional
            DESCRIPTION. The default is None.

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

    def __getitem__(self, i):
        """.

        Parameters
        ----------
        i : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        assert i < len(self)
        return ee.Image(self.imagery_list.get(i))

    def __len__(self):
        """.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.size


class DSMImagery(CollectionClass):
    """."""

    def get_bands(self, date, geometry):
        """.

        Parameters
        ----------
        date : TYPE
            DESCRIPTION.
        geometry : TYPE
            DESCRIPTION.
        scale : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        images = self.imagery.filterBounds(geometry)

        def add_hour_difference(img):
            return img.setMulti({"hour_difference": date.difference(img.date(),
                                                                    'hour')})

        images = images.map(add_hour_difference)
        images = images.filter(ee.Filter.gt('hour_difference', 0))
        images = images.sort('hour_difference', True).reduce(
                    ee.Reducer.firstNonNull()).select(["%s_first" % b
                                                       for b in self.bands],
                                                      self.bands)
        return images.clip(geometry)


class WindImagery(CollectionClass):
    """."""

    def __init__(self, link, bands=None, before_range=(-12, 'hour')):
        """.

        Parameters
        ----------
        link : TYPE
            DESCRIPTION.
        bands : TYPE, optional
            DESCRIPTION. The default is None.
        before_range : TYPE, optional
            DESCRIPTION. The default is (-12, 'hour').

        Returns
        -------
        None.

        """
        super(WindImagery, self).__init__(link, bands)
        self.before = before_range

    def get_bands(self, date, geometry):
        """.

        Parameters
        ----------
        date : TYPE
            DESCRIPTION.
        geometry : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        images = self.get_clipped_bands(geometry)
        images = images.filter(ee.Filter.date(date.advance(*self.before), date)
                               )
        images = images.map(lambda img: self.stack_bands(img, date).clip(
            geometry))

        return merge_bands(images)

    def stack_bands(self, img, date):
        """.

        Parameters
        ----------
        img : TYPE
            DESCRIPTION.
        date : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return img.select(self.bands,
                          [ee.Algorithms.String(date.difference(img.date(),
                                                                'hour').int()
                                                ).cat('_%s' % b)
                           for b in self.bands])


def merge_bands(images):
    """.

    Parameters
    ----------
    images : ee.ImageCollection
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return ee.Image(images.iterate(
                (lambda image, previous: ee.Image(previous).addBands(image)),
                ee.Image(None)))
