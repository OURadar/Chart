# Atlas overlay

import os
import json
import pandas
import zipfile
import shapely
import geopandas
import matplotlib
import matplotlib.patheffects
import urllib.request
import numpy as np

from . import base
from . import font

folder = '{}/shapefiles'.format(base.storageHome)

r_earth = 6378.0
deg2rad = np.pi / 180.0
rad2deg = 1.0 / deg2rad
ring_color = '#78dcff'
weight_big = 50000
weight_med = 95000      # Norman is 95694

def showDebugMessages():
    base.showDebugMessages()

def showInfoMessages():
    base.showInfoMessages()

def countryFromCoord(coord=(-97.46381, 35.23682)):
    db = [
        [1.601554, 42.546245, 'Andorra'],
        [53.847818, 23.424076, 'United Arab Emirates'],
        [67.709953, 33.93911, 'Afghanistan'],
        [-61.796428, 17.060816, 'Antigua and Barbuda'],
        [-63.068615, 18.220554, 'Anguilla'],
        [20.168331, 41.153332, 'Albania'],
        [45.038189, 40.069099, 'Armenia'],
        [-69.060087, 12.226079, 'Netherlands Antilles'],
        [17.873887, -11.202692, 'Angola'],
        [-0.071389, -75.250973, 'Antarctica'],
        [-63.616672, -38.416097, 'Argentina'],
        [-170.132217, -14.270972, 'American Samoa'],
        [14.550072, 47.516231, 'Austria'],
        [133.775136, -25.274398, 'Australia'],
        [-69.968338, 12.52111, 'Aruba'],
        [47.576927, 40.143105, 'Azerbaijan'],
        [17.679076, 43.915886, 'Bosnia and Herzegovina'],
        [-59.543198, 13.193887, 'Barbados'],
        [90.356331, 23.684994, 'Bangladesh'],
        [4.469936, 50.503887, 'Belgium'],
        [-1.561593, 12.238333, 'Burkina Faso'],
        [25.48583, 42.733883, 'Bulgaria'],
        [50.637772, 25.930414, 'Bahrain'],
        [29.918886, -3.373056, 'Burundi'],
        [2.315834, 9.30769, 'Benin'],
        [-64.75737, 32.321384, 'Bermuda'],
        [114.727669, 4.535277, 'Brunei'],
        [-63.588653, -16.290154, 'Bolivia'],
        [-51.92528, -14.235004, 'Brazil'],
        [-77.39628, 25.03428, 'Bahamas'],
        [90.433601, 27.514162, 'Bhutan'],
        [3.413194, -54.423199, 'Bouvet Island'],
        [24.684866, -22.328474, 'Botswana'],
        [27.953389, 53.709807, 'Belarus'],
        [-88.49765, 17.189877, 'Belize'],
        [-106.346771, 56.130366, 'Canada'],
        [96.870956, -12.164165, 'Cocos [Keeling] Islands'],
        [21.758664, -4.038333, 'Congo [DRC]'],
        [20.939444, 6.611111, 'Central African Republic'],
        [15.827659, -0.228021, 'Congo [Republic]'],
        [8.227512, 46.818188, 'Switzerland'],
        [-5.54708, 7.539989, "C™te d'Ivoire"],
        [-159.777671, -21.236736, 'Cook Islands'],
        [-71.542969, -35.675147, 'Chile'],
        [12.354722, 7.369722, 'Cameroon'],
        [104.195397, 35.86166, 'China'],
        [-74.297333, 4.570868, 'Colombia'],
        [-83.753428, 9.748917, 'Costa Rica'],
        [-77.781167, 21.521757, 'Cuba'],
        [-24.013197, 16.002082, 'Cape Verde'],
        [105.690449, -10.447525, 'Christmas Island'],
        [33.429859, 35.126413, 'Cyprus'],
        [15.472962, 49.817492, 'Czech Republic'],
        [10.451526, 51.165691, 'Germany'],
        [42.590275, 11.825138, 'Djibouti'],
        [9.501785, 56.26392, 'Denmark'],
        [-61.370976, 15.414999, 'Dominica'],
        [-70.162651, 18.735693, 'Dominican Republic'],
        [1.659626, 28.033886, 'Algeria'],
        [-78.183406, -1.831239, 'Ecuador'],
        [25.013607, 58.595272, 'Estonia'],
        [30.802498, 26.820553, 'Egypt'],
        [-12.885834, 24.215527, 'Western Sahara'],
        [39.782334, 15.179384, 'Eritrea'],
        [-3.74922, 40.463667, 'Spain'],
        [40.489673, 9.145, 'Ethiopia'],
        [25.748151, 61.92411, 'Finland'],
        [179.414413, -16.578193, 'Fiji'],
        [-59.523613, -51.796253, 'Falkland Islands'],
        [150.550812, 7.425554, 'Micronesia'],
        [-6.911806, 61.892635, 'Faroe Islands'],
        [2.213749, 46.227638, 'France'],
        [11.609444, -0.803689, 'Gabon'],
        [-3.435973, 55.378051, 'United Kingdom'],
        [-61.604171, 12.262776, 'Grenada'],
        [43.356892, 42.315407, 'Georgia'],
        [-53.125782, 3.933889, 'French Guiana'],
        [-2.585278, 49.465691, 'Guernsey'],
        [-1.023194, 7.946527, 'Ghana'],
        [-5.345374, 36.137741, 'Gibraltar'],
        [-42.604303, 71.706936, 'Greenland'],
        [-15.310139, 13.443182, 'Gambia'],
        [-9.696645, 9.945587, 'Guinea'],
        [-62.067641, 16.995971, 'Guadeloupe'],
        [10.267895, 1.650801, 'Equatorial Guinea'],
        [21.824312, 39.074208, 'Greece'],
        [-36.587909, -54.429579, 'South Georgia and the South Sandwich Islands'],
        [-90.230759, 15.783471, 'Guatemala'],
        [144.793731, 13.444304, 'Guam'],
        [-15.180413, 11.803749, 'Guinea-Bissau'],
        [-58.93018, 4.860416, 'Guyana'],
        [34.308825, 31.354676, 'Gaza Strip'],
        [114.109497, 22.396428, 'Hong Kong'],
        [73.504158, -53.08181, 'Heard Island and McDonald Islands'],
        [-86.241905, 15.199999, 'Honduras'],
        [15.2, 45.1, 'Croatia'],
        [-72.285215, 18.971187, 'Haiti'],
        [19.503304, 47.162494, 'Hungary'],
        [113.921327, -0.789275, 'Indonesia'],
        [-8.24389, 53.41291, 'Ireland'],
        [34.851612, 31.046051, 'Israel'],
        [-4.548056, 54.236107, 'Isle of Man'],
        [78.96288, 20.593684, 'India'],
        [71.876519, -6.343194, 'British Indian Ocean Territory'],
        [43.679291, 33.223191, 'Iraq'],
        [53.688046, 32.427908, 'Iran'],
        [-19.020835, 64.963051, 'Iceland'],
        [12.56738, 41.87194, 'Italy'],
        [-2.13125, 49.214439, 'Jersey'],
        [-77.297508, 18.109581, 'Jamaica'],
        [36.238414, 30.585164, 'Jordan'],
        [138.252924, 36.204824, 'Japan'],
        [37.906193, -0.023559, 'Kenya'],
        [74.766098, 41.20438, 'Kyrgyzstan'],
        [104.990963, 12.565679, 'Cambodia'],
        [-168.734039, -3.370417, 'Kiribati'],
        [43.872219, -11.875001, 'Comoros'],
        [-62.782998, 17.357822, 'Saint Kitts and Nevis'],
        [127.510093, 40.339852, 'North Korea'],
        [127.766922, 35.907757, 'South Korea'],
        [47.481766, 29.31166, 'Kuwait'],
        [-80.566956, 19.513469, 'Cayman Islands'],
        [66.923684, 48.019573, 'Kazakhstan'],
        [102.495496, 19.85627, 'Laos'],
        [35.862285, 33.854721, 'Lebanon'],
        [-60.978893, 13.909444, 'Saint Lucia'],
        [9.555373, 47.166, 'Liechtenstein'],
        [80.771797, 7.873054, 'Sri Lanka'],
        [-9.429499, 6.428055, 'Liberia'],
        [28.233608, -29.609988, 'Lesotho'],
        [23.881275, 55.169438, 'Lithuania'],
        [6.129583, 49.815273, 'Luxembourg'],
        [24.603189, 56.879635, 'Latvia'],
        [17.228331, 26.3351, 'Libya'],
        [-7.09262, 31.791702, 'Morocco'],
        [7.412841, 43.750298, 'Monaco'],
        [28.369885, 47.411631, 'Moldova'],
        [19.37439, 42.708678, 'Montenegro'],
        [46.869107, -18.766947, 'Madagascar'],
        [171.184478, 7.131474, 'Marshall Islands'],
        [21.745275, 41.608635, 'Macedonia'],
        [-3.996166, 17.570692, 'Mali'],
        [95.956223, 21.913965, 'Myanmar'],
        [103.846656, 46.862496, 'Mongolia'],
        [113.543873, 22.198745, 'Macau'],
        [145.38469, 17.33083, 'Northern Mariana Islands'],
        [-61.024174, 14.641528, 'Martinique'],
        [-10.940835, 21.00789, 'Mauritania'],
        [-62.187366, 16.742498, 'Montserrat'],
        [14.375416, 35.937496, 'Malta'],
        [57.552152, -20.348404, 'Mauritius'],
        [73.22068, 3.202778, 'Maldives'],
        [34.301525, -13.254308, 'Malawi'],
        [-102.552784, 23.634501, 'Mexico'],
        [101.975766, 4.210484, 'Malaysia'],
        [35.529562, -18.665695, 'Mozambique'],
        [18.49041, -22.95764, 'Namibia'],
        [165.618042, -20.904305, 'New Caledonia'],
        [8.081666, 17.607789, 'Niger'],
        [167.954712, -29.040835, 'Norfolk Island'],
        [8.675277, 9.081999, 'Nigeria'],
        [-85.207229, 12.865416, 'Nicaragua'],
        [5.291266, 52.132633, 'Netherlands'],
        [8.468946, 60.472024, 'Norway'],
        [84.124008, 28.394857, 'Nepal'],
        [166.931503, -0.522778, 'Nauru'],
        [-169.867233, -19.054445, 'Niue'],
        [174.885971, -40.900557, 'New Zealand'],
        [55.923255, 21.512583, 'Oman'],
        [-80.782127, 8.537981, 'Panama'],
        [-75.015152, -9.189967, 'Peru'],
        [-149.406843, -17.679742, 'French Polynesia'],
        [143.95555, -6.314993, 'Papua New Guinea'],
        [121.774017, 12.879721, 'Philippines'],
        [69.345116, 30.375321, 'Pakistan'],
        [19.145136, 51.919438, 'Poland'],
        [-56.27111, 46.941936, 'Saint Pierre and Miquelon'],
        [-127.439308, -24.703615, 'Pitcairn Islands'],
        [-66.590149, 18.220833, 'Puerto Rico'],
        [35.233154, 31.952162, 'Palestinian Territories'],
        [-8.224454, 39.399872, 'Portugal'],
        [134.58252, 7.51498, 'Palau'],
        [-58.443832, -23.442503, 'Paraguay'],
        [51.183884, 25.354826, 'Qatar'],
        [55.536384, -21.115141, 'RŽunion'],
        [24.96676, 45.943161, 'Romania'],
        [21.005859, 44.016521, 'Serbia'],
        [105.318756, 61.52401, 'Russia'],
        [29.873888, -1.940278, 'Rwanda'],
        [45.079162, 23.885942, 'Saudi Arabia'],
        [160.156194, -9.64571, 'Solomon Islands'],
        [55.491977, -4.679574, 'Seychelles'],
        [30.217636, 12.862807, 'Sudan'],
        [18.643501, 60.128161, 'Sweden'],
        [103.819836, 1.352083, 'Singapore'],
        [-10.030696, -24.143474, 'Saint Helena'],
        [14.995463, 46.151241, 'Slovenia'],
        [23.670272, 77.553604, 'Svalbard and Jan Mayen'],
        [19.699024, 48.669026, 'Slovakia'],
        [-11.779889, 8.460555, 'Sierra Leone'],
        [12.457777, 43.94236, 'San Marino'],
        [-14.452362, 14.497401, 'Senegal'],
        [46.199616, 5.152149, 'Somalia'],
        [-56.027783, 3.919305, 'Suriname'],
        [6.613081, 0.18636, 'Sao Tome and Principe'],
        [-88.89653, 13.794185, 'El Salvador'],
        [38.996815, 34.802075, 'Syria'],
        [31.465866, -26.522503, 'Swaziland'],
        [-71.797928, 21.694025, 'Turks and Caicos Islands'],
        [18.732207, 15.454166, 'Chad'],
        [69.348557, -49.280366, 'French Southern Territories'],
        [0.824782, 8.619543, 'Togo'],
        [100.992541, 15.870032, 'Thailand'],
        [71.276093, 38.861034, 'Tajikistan'],
        [-171.855881, -8.967363, 'Tokelau'],
        [125.727539, -8.874217, 'Timor-Leste'],
        [59.556278, 38.969719, 'Turkmenistan'],
        [9.537499, 33.886917, 'Tunisia'],
        [-175.198242, -21.178986, 'Tonga'],
        [35.243322, 38.963745, 'Turkey'],
        [-61.222503, 10.691803, 'Trinidad and Tobago'],
        [177.64933, -7.109535, 'Tuvalu'],
        [120.960515, 23.69781, 'Taiwan'],
        [34.888822, -6.369028, 'Tanzania'],
        [31.16558, 48.379433, 'Ukraine'],
        [32.290275, 1.373333, 'Uganda'],
        [-95.712891, 37.09024, 'United States'],
        [-55.765835, -32.522779, 'Uruguay'],
        [64.585262, 41.377491, 'Uzbekistan'],
        [12.453389, 41.902916, 'Vatican City'],
        [-61.287228, 12.984305, 'Saint Vincent and the Grenadines'],
        [-66.58973, 6.42375, 'Venezuela'],
        [-64.639968, 18.420695, 'British Virgin Islands'],
        [-64.896335, 18.335765, 'U.S. Virgin Islands'],
        [108.277199, 14.058324, 'Vietnam'],
        [166.959158, -15.376706, 'Vanuatu'],
        [-177.156097, -13.768752, 'Wallis and Futuna'],
        [-172.104629, -13.759029, 'Samoa'],
        [20.902977, 42.602636, 'Kosovo'],
        [48.516388, 15.552727, 'Yemen'],
        [45.166244, -12.8275, 'Mayotte'],
        [22.937506, -30.559482, 'South Africa'],
        [27.849332, -13.133897, 'Zambia'],
        [29.154857, -19.015438, 'Zimbabwe']
    ]
    dmin = 9999.0
    country = 'Unknown'
    for x in db:
        d = (coord[0] - x[0]) ** 2 + (coord[1] - x[1]) ** 2
        if dmin > d:
            dmin = d
            country = x[2]
            #print('d = {} --> {}'.format(d, country))
    return country

def makeRotationX(phi):
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), np.sin(phi)], [0.0, -np.sin(phi), np.cos(phi)]])

def makeRotationY(phi):
    return np.array([[np.cos(phi), 0.0, -np.sin(phi)], [0.0, 1.0, 0.0], [np.sin(phi), 0.0, np.cos(phi)]])

def makeRotationZ(phi):
    return np.array([[np.cos(phi), np.sin(phi), 0.0], [-np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])

def makeRotationForCoord(lon=-97.46381, lat=35.23682):
    return np.matmul(makeRotationY(-lon * deg2rad), makeRotationX(lat * deg2rad))

def makeViewBox(xmin, xmax, ymin, ymax, lon=-97.46381, lat=35.23682):
    # A local function to wrap once the supplied longitude to +/- 180.0
    def wrap(lon):
        if lon > 180.0:
            lon -= 360.0
        elif lon < -180.0:
            lon += 360.0
        return lon
    lonmin = wrap(lon + xmin / (r_earth * np.cos(lat * deg2rad)) * rad2deg)
    lonmax = wrap(lon + xmax / (r_earth * np.cos(lat * deg2rad)) * rad2deg)
    latmin = max(-90.0, lat + ymin / r_earth * rad2deg)
    latmax = min(+90.0, lat + ymax / r_earth * rad2deg)
    return shapely.geometry.box(lonmin, lonmax, latmin, latmax)

def getShapefiles(country='United States', force_download=False, cache=True, savecache=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    subfolder = '{}/{}'.format(folder, country)
    base.logger.debug('country = {}   subfolder = {}'.format(country, subfolder))
    if not os.path.exists(subfolder):
        base.logger.debug('country = {} does not exist'.format(country))
        if country == '__user__':
            base.logger.debug('No user maps. Early return')
            return None
        os.makedirs(subfolder)

    # Fullpath of the index
    file = '{}/index'.format(subfolder)
    base.logger.debug('index file = {}'.format(file))
    if not os.path.isfile(file) or force_download:
        url = 'https://arrc.ou.edu/iradar/shapefiles/{}/index'.format(country.replace(' ', '%20'))
        base.logger.info('Maps of {} not exist. Downloading from server ...'.format(country))
        base.logger.info('URL = '.format(url))
        urllib.request.urlretrieve(url, file)

    # Read in the index file
    with open(file, 'r') as file:
        data = ''.join(file.read().strip())
    shapes = json.loads(data)

    # Check individual shapes
    for shape in shapes:
        file = '{}/{}.shp'.format(subfolder, shape['file'])
        base.logger.debug('shapefile = {}'.format(file))
        if not os.path.isfile(file) or force_download:
            url = 'https://arrc.ou.edu/iradar/shapefiles/{}/{}.zip'.format(country.replace(' ', '%20'), shape['file'])
            file = '{}/{}/{}.zip'.format(folder, country, shape['file'])
            base.logger.info('Downloading maps from server ...')
            base.logger.info('{}'.format(url))
            base.logger.info('{}'.format(file))
            if not os.path.isfile(file) or force_download:
                urllib.request.urlretrieve(url, file)
            with zipfile.ZipFile(file) as zipped:
                for info in zipped.infolist():
                    file = '{}/{}/{}'.format(folder, country, info.filename)
                    base.logger.info('Unzipping file {} ...'.format(info.filename))
                    base.logger.debug('{}'.format(file))
                    with open(file, 'wb') as outfile:
                        with zipped.open(info) as zippedfile:
                            outfile.write(zippedfile.read())
        # The shapefile
        file = '{}/{}/{}.shp'.format(folder, country, shape['file'])
        file4log = base.shortenPath(file, 4)
        cachefile = '{}/{}/{}.pkl'.format(folder, country, shape['file'])
        cachefile4log = base.shortenPath(cachefile, 4)
        if os.path.isfile(cachefile) and cache:
            base.logger.debug('Loading from cache {} ...'.format(cachefile4log))
            frame = pandas.read_pickle(cachefile)
        else:
            base.logger.debug('Loading {} ...'.format(file4log))
            frame = geopandas.read_file(file)
            if shape['type'] == 'label':
                # Make a new dataframe with (label, sort) at columns (0, 1), respectively
                frame = frame.iloc[:, [shape['label'], shape['sort'], frame.columns.get_loc('geometry')]]
            else:
                frame = frame.loc[:, ['geometry']]
            if savecache:
                base.logger.info('Saving cache as {} ...'.format(cachefile4log))
                frame.to_pickle(cachefile)
        shape['dataframe'] = frame
    return shapes

class Layer:
    # Type:
    #   - polygon (shapefile)
    #   - label (shapefile)
    #   - line (update has no effects)
    #   - text (update consolidates from sources)
    def __init__(self, parent, dataframe=None, type='line', color='#888888', linewidth=1.0):
        self.parent = parent
        self.dataframe = dataframe
        self.type = type
        self.color = color
        self.linewidth = linewidth
        self.viewbox = shapely.geometry.box(-180.0, 180.0, -90.0, 90.0)
        self.zorder = 0.0
        self.s200 = 10000
        self.s500 = 50000
        self.s1000 = 100000
        self.needsUpdate = True
        self.fontProperties = None
        self.featureScale = 1.0

    # Extract and project the geometry to the view box
    def update(self):
        if not self.needsUpdate:
            return
        # A local function to project (lon, lat) coordinates from shapefiles onto the viewbox Cartesian coordinate (x, y)
        def project(coords):
            m = r_earth * np.cos(coords[:, 1])
            y = r_earth * np.sin(coords[:, 1])
            z = m * np.cos(coords[:, 0])
            x = m * np.sin(coords[:, 0])
            p = np.array((x, y, z)).transpose()
            p = np.matmul(p, self.rotation)
            return p
        if self.type == 'polygon':
            self.lines = []
            for k, row in self.dataframe.iterrows():
                item = row['geometry']
                domain = shapely.geometry.box(*item.bounds)
                if not self.viewbox.intersects(domain):
                    continue
                if item.geom_type == 'MultiPolygon':
                    for subitem in item.geoms:
                        subdomain = shapely.geometry.box(*subitem.bounds)
                        if not self.viewbox.intersects(subdomain):
                            continue
                        line = project(np.array(subitem.exterior.coords) * deg2rad)
                        self.lines.append(line)
                elif item.geom_type == 'Polygon':
                    line = project(np.array(item.exterior.coords) * deg2rad)
                    self.lines.append(line)
                elif item.geom_type == 'MultiLineString':
                    for subitem in item.geoms:
                        subdomain = shapely.geometry.box(*subitem.bounds)
                        if not self.viewbox.intersects(subdomain):
                            continue
                        line = project(np.array(subitem.coords) * deg2rad)
                        self.lines.append(line)
                elif item.geom_type == 'LineString':
                    line = project(np.array(item.coords) * deg2rad)
                    self.lines.append(line)
                else:
                    base.logger.debug('Type {} skipped.'.format(item.geom_type))
            self.needsUpdate = False
        elif self.type == 'label':
            # Convert dataframe to local array of labels
            points = project(self.coords)
            points[:, 2] = points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]
            texts = [(p[0], p[1], p[2], q[0], q[1], 'w') for p, q in zip(points, self.dataframe.iloc[:, [0, 1]].values)]
            # First-level masking based on s200, s500 & s1000 thresholds to sort values
            s = self.dataframe.iloc[:, 1].values
            r = np.sqrt(points[:, 2])
            mask = (r < 100.0)
            mask = np.logical_or(mask, np.logical_and(r < 200.0, s > self.s200))
            mask = np.logical_or(mask, np.logical_and(r < 500.0, s > self.s500))
            mask = np.logical_or(mask, np.logical_and(r < 1000.0, s > self.s1000))
            self.texts = [d for d, s in zip(texts, mask) if s]
            self.needsUpdate = False
        elif self.type == 'text':
            # Consolidate labels & text
            texts = []
            for layer in self.sources:
                texts += layer.texts
            # Sort the array based on sort values
            texts.sort(key=lambda x: x[4], reverse=True)
            # Now it is ready to be tagged on the layer data
            self.texts = texts
            self.needsUpdate = False
            base.logger.debug('texts has {} elements'.format(len(self.texts)))
        elif self.type == 'line':
            self.needsUpdate = False

    def draw(self, ax):
        if self.needsUpdate:
            self.update()
        linewidth = self.featureScale * self.linewidth
        outlinewidth = self.featureScale * 2.5
        if self.type == 'polygon' or self.type == 'line':
            for points in self.lines:
                line = matplotlib.lines.Line2D(points[:, 0], points[:, 1], color=self.color, linewidth=linewidth, zorder=self.zorder)
                ax.add_line(line)
        elif self.type == 'label' or self.type == 'text':
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            points_per_km = 0.5 * (ax.bbox.bounds[2] / (xmax - xmin) +
                                   ax.bbox.bounds[3] / (ymax - ymin))
            if self.fontProperties is None:
                self.fontProperties = font.Properties(scale=self.featureScale)
            # Draw and cull labels outside of the domain
            extents = []
            for x, y, s, label, weight, color in self.texts:
                if x < xmin or x > xmax or y < ymin or y > ymax:
                    continue
                p = 1.0 if weight < 1.0 else 2500.0 * np.log(weight) / weight
                if p > points_per_km and s > 100.0:
                    continue
                if weight > weight_big:
                    prop = self.fontProperties.label_big
                elif weight > weight_med:
                    prop = self.fontProperties.label_medium
                else:
                    prop = self.fontProperties.label_small
                text = matplotlib.text.Text(x, y, label,
                                            ha='center',
                                            va='center',
                                            color=color,
                                            zorder=self.zorder,
                                            fontproperties=prop)
                ax.add_artist(text)
                text_box = text.get_window_extent(ax.figure.canvas.renderer)
                # Check against previous label, cull if the space is preoccupied
                overlap = False
                for extent in extents:
                    if extent.overlaps(text_box):
                        overlap = True
                        break
                if overlap:
                    text.remove()
                else:
                    text.set_path_effects([
                        matplotlib.patheffects.Stroke(linewidth=outlinewidth, foreground=(0.0, 0.0, 0.0, 0.6)),
                        matplotlib.patheffects.Normal()
                    ])
                    extents.append(text_box)
        else:
            base.logger.warning('Unexpected type {}'.format(self.type))

    def initAsRings(self, radii):
        self.type = 'line'
        self.color = '#78dcff'
        self.linewidth = 1.5
        self.zorder = 40.0
        self.needsUpdate = False
        self.lines = []
        a = np.linspace(1.25 * np.pi, 3.248 * np.pi, 180)
        p = np.array([np.cos(a), np.sin(a)]).transpose((1, 0))
        for r in radii:
            self.lines.append(r * p)
        return self

    def initAsRingLabels(self, radii):
        self.type = 'text'
        self.color = '#78dcff'
        self.linewidth = 1.0
        self.zorder = 41.0
        self.needsUpdate = False
        self.texts = []
        # Ring labels at +45-deg and -45-deg and they get s500 threshold as sort value
        xx = radii * np.cos(0.25 * np.pi)
        yy = radii * np.sin(0.25 * np.pi)
        ss = radii * radii
        for x, y, s, r in zip(xx, yy, ss, radii):
            if r <= 1.0:
                continue
            label = '{:.0f} km'.format(r)
            self.texts.append((+x, +y, s, label, self.s200, ring_color))
            self.texts.append((-x, -y, s, label, self.s200, ring_color))
        return self

    def initAsRHIGrids(self, radii):
        self.type = 'line'
        self.color = '#78dcff'
        self.linewidth = 1.5
        self.zorder = 40.0
        self.needsUpdate = False
        self.lines = []
        for r in radii:
            self.lines.append(np.array([[r,0], [r,100]]))
        for r in np.arange(0,20,5):
            self.lines.append(np.array([[-100,r], [100,r]]))
        return self

    def initAsRHILabels(self, radii):
        self.type = 'text'
        self.color = '#78dcff'
        self.linewidth = 1.0
        self.zorder = 41.0
        self.needsUpdate = False
        self.texts = []
        # Ring labels at +45-deg and -45-deg and they get s500 threshold as sort value
        # xx = radii * np.cos(0.25 * np.pi)
        # yy = radii * np.sin(0.25 * np.pi)
        # ss = radii * radii
        for r in radii:
            if r <= 1.0:
                continue
            label = '{:.0f} km'.format(r)
            self.texts.append((+r, -0.5, r * r, label, self.s200, ring_color))
            self.texts.append((-r, -0.5, r * r, label, self.s200, ring_color))
            self.texts.append((0, +r, r * r, label, self.s200, ring_color))
        return self

    def initAsTexts(self):
        self.type = 'text'
        self.color = 'varying'
        self.linewidth = 1.0
        # Answer to the Ultimate Question of Life, the Universe, and Everything
        self.zorder = 42.0
        self.needsUpdate = False
        # Source layers to generate a consolidated set
        self.sources = []
        # An internal array that holds (x, y, r ** 2, label, sort_value, color)
        self.labels = []
        return self

    def appendText(self, x, y, s, label, sort, color):
        self.texts.append((x, y, s, label, self.s500, ring_color))

class Overlay:
    def __init__(self, origin=(-97.46381, 35.23682), featureScale=1.0, scantype='PPI'):
        self.lon = 0.0
        self.lat = 0.0
        self.scantype = scantype
        self.filenames = []
        self.ringRadii = np.concatenate(([1.0], np.arange(30.0, 250.0, 30.0)))
        # Layers are units that hold the original dataframe, the geometry ready to be drawn
        # Layer 0 - text - special layer, conslidated text (only one)
        # Layer 1 - line - concentric range rings
        # Layer 2 - label - labels of the concentric rings
        # Layer 3+ - polygon / label - layers constructed based on shapefiles
        self.layers = [Layer(self).initAsTexts(),
                       Layer(self).initAsRings(self.ringRadii),
                       Layer(self).initAsRingLabels(self.ringRadii)]
        self.layers[0].sources.append(self.layers[2])
        self.layers[0].needsUpdate = True
        self.featureScale = featureScale
        self.setOrigin(origin)
        if not ((self.scantype == 'PPI') or (self.scantype == 'RHI')):
            raise ChartError("Unrecognized ScanType.")

    def __repr__(self):
        string = 'lon = {}\nlat = {}\ncountry = {}\n'.format(self.lon, self.lat, self.country)
        string += 'Layers:\n'
        string += '   {:8s}    {:7s}    {:9s}    {:5s}\n'.format('type', 'color', 'linewidth', 'zorder')
        for layer in self.layers:
            string += '   {:8s}    {:7s}    {:9.2f}    {:.0f}\n'.format(layer.type, layer.color, layer.linewidth, layer.zorder)
        return string

    def setOrigin(self, origin):
        self.lon = origin[0]
        self.lat = origin[1]
        self.country = countryFromCoord((self.lon, self.lat))
        self.rotation = makeRotationForCoord(self.lon, self.lat)
        if len(self.layers) > 3:
            self._prepare()

    def setFeatureScale(self, featureScale):
        self.featureScale = featureScale
        self.layers[0].needsUpdate = True
        for layer in self.layers[3:]:
            layer.featureScale = self.featureScale

    # Load the shapefiles
    def load(self):
        if len(self.layers) > 3:
            base.logger.warning('Layers already loaded.  (count = {})'.format(len(self.layers)))
            return
        shapes = getShapefiles(self.country)
        userShapes = getShapefiles('__user__')
        if userShapes:
            base.logger.info('User shapes exist with {} layers'.format(len(userShapes)))
            for shape in userShapes:
                shapes.append(shape)
        base.logger.debug('shapes - {} layers'.format(len(shapes)))

        for k, shape in enumerate(shapes):
            layer = Layer(self, shape['dataframe'], shape['type'])
            if 'color' in shape:
                layer.color = '#' + shape['color']
            if 'linewidth' in shape:
                layer.linewidth = shape['linewidth']
            if shape['type'] == 'label':
                layer.s200 = shape['s200'] if 's200' in shape else 10000
                layer.s500 = shape['s500'] if 's500' in shape else 50000
                layer.s1000 = shape['s1000'] if 's1000' in shape else 100000
                layer.coords = np.array([p.coords[0] for p in layer.dataframe.loc[:, 'geometry']]) * deg2rad
                self.layers[0].sources.append(layer)
            layer.zorder = float(k)
            self.layers.append(layer)

    # Ask the layers to extract and project the elements within the view box
    def _prepare(self):
        # Load the shapefiles if there is only 3 initial layers
        if len(self.layers) == 3:
            self.load()
        # Approximate (lon, lat) domain
        self.viewbox = makeViewBox(self.xmin, self.xmax, self.ymin, self.ymax, self.lon, self.lat)
        base.logger.debug('coord = {:.4f} {:.4f} --> domain = [{:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(
            self.lon, self.lat, self.viewbox.bounds[0], self.viewbox.bounds[1], self.viewbox.bounds[2], self.viewbox.bounds[3]))
        if self.scantype == 'PPI':
            self.layers[1].initAsRings(self.ringRadii)
            self.layers[2].initAsRingLabels(self.ringRadii)
        elif self.scantype == 'RHI':
            self.layers[1].initAsRHIGrids(self.ringRadii)
            self.layers[2].initAsRHILabels(self.ringRadii)
        for layer in self.layers[3:]:
            layer.viewbox = self.viewbox
            layer.rotation = self.rotation
            layer.featureScale = self.featureScale
            layer.update()
        base.logger.debug('Updating layers[0] ...')
        self.layers[0].featureScale = self.featureScale
        self.layers[0].update()

    # Prepare (extract and project) the geometry acoording to the axes limits
    def prepareForAxes(self, ax):
        base.logger.debug('Overlay.prepareForAxes')
        self.xmin, self.xmax = ax.get_xlim()
        self.ymin, self.ymax = ax.get_ylim()
        self._prepare()

    # Prepare (extract and project) the geometry acoording to the supplied rectangular
    def prepareForRect(self, rect):
        self.xmin, self.ymin = rect[0], rect[1]
        self.xmax, self.ymax = rect[0] + rect[2], rect[1] + rect[3]
        self._prepare()

    # Set the radii of concentric range rings
    def setRingRadii(self, radii):
        self.ringRadii = np.array(radii)
        self.layers[0].needsUpdate = True

    def draw(self, ax):
        base.logger.debug('self.layers[0].needsUpdate: = {}'.format(self.layers[0].needsUpdate))
        if self.layers[0].needsUpdate:
            self.prepareForAxes(ax)
        # Local function to draw a line and add it to an axis
        if self.scantype == 'PPI':
            for layer in self.layers:
                if (layer.type == 'polygon' or layer.type == 'line'):
                    layer.draw(ax)
            self.layers[0].draw(ax)
        elif self.scantype == 'RHI':
            for layer in self.layers[1:3]:
                layer.draw(ax)
