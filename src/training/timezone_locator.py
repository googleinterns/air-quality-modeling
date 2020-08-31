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

#  pip install timezonefinder[numba] # also installs numba -> x100 speedup
from timezonefinder import TimezoneFinderL
from pytz import timezone, utc
from datetime import datetime

class TimeLocator:
    """Converts UTC time to local time of latitude and longitude positions."""

    def __init__(self):
        """Initializes TimeLocator class."""
        self.tf = TimezoneFinderL(in_memory=True)

    def locate(self, lat, lon, hod, dom, moy):
        """Converts given time to local time of latitude and longitude.

        Parameters
        ----------
        lat : float
            latitude
        lon : float
            longitude
        hod : hour of day
            int
        dom : int
            day of month
        moy : int
            month of year

        Returns
        -------
        t_hod : int
            local hour of day
        t_dow : int
            local day of week
        t_dom : int
            local day of month
        t_moy : int
            local month of year

        """
        date = datetime(year=2019, month=moy, day=dom, hour=hod, tzinfo=utc)
        target_timezone = timezone(self.tf.timezone_at(lng=lon, lat=lat))
        target_date = date.astimezone(target_timezone)
        t_hod, t_dow = target_date.hour, target_date.weekday()
        t_dom, t_moy = target_date.day, target_date.month
        return t_hod, t_dow, t_dom, t_moy
