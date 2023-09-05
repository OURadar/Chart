import numpy as np
from netCDF4 import Dataset
from . import atlas
from . import chart
import datetime
class RadarKit:
    def __init__(self, file):
        with Dataset(file, mode='r') as nc:
            self.name = nc.getncattr('TypeName')
            self.elev = np.array(nc.variables['Elevation'][:], dtype=np.float32)
            self.azim = np.array(nc.variables['Azimuth'][:], dtype=np.float32)
            self.gatewidth = np.array(nc.variables['GateWidth'][:], dtype=np.float32)
            self.values = np.array(nc.variables[self.name][:], dtype=np.float32)
            self.values[self.values < -90] = np.nan
            self.longitude = nc.getncattr('Longitude')
            self.latitude = nc.getncattr('Latitude')
            self.height = nc.getncattr('Height')
            self.sweepElev = nc.getncattr('Elevation')
            self.sweepAz = nc.getncattr('Azimuth')
            self.sweepTime = nc.getncattr('Time')
            self.scantype = nc.getncattr('ScanType')
            self.symbol = file.split('.')[-2].split('-')[-1]
    def genOverlay(self):
        return atlas.Overlay((self.longitude, self.latitude),scantype = self.scantype)

    def Image(self,**kwargs):
        r = 1.0e-3 * np.arange(self.values.shape[1]) * self.gatewidth[0]
        a = np.deg2rad(self.azim)
        e = np.deg2rad(self.elev)
        t = datetime.datetime.utcfromtimestamp(self.sweepTime)
        timestr = t.strftime('%Y/%m/%d %H:%M:%S')
        if self.scantype =='PPI':
            title = f'{timestr} UTC  EL: {self.sweepElev:.2f}°'
        elif self.scantype =='RHI':
            title = f'{timestr} UTC  AZ: {self.sweepAz:.2f}°'
        if not('title' in kwargs):
            kwargs.update({'title':title})
        return chart.Image(e, a, r, self.values, style=self.symbol, figsize=(800, 600), maxrange=30.0, scantype = self.scantype,**kwargs)