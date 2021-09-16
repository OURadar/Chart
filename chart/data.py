# Example dataset

import os
import sys
import glob
import zipfile
import datetime
import urllib.request
import numpy as np

from . import base

folder = '{}/data'.format(base.storageHome)

prefix = 'PX-20170220-050706-E2.4'

# np.set_printoptions(precision=2)

def check(verbose=1):
    if not os.path.exists(folder):
        base.logger.info('Making folder {} ...'.format(folder))
        os.makedirs(folder)

    files = ['{}-{}.nc'.format(prefix, x) for x in ['Z', 'V', 'W', 'D', 'P', 'R']]

    allExist = True
    for file in files:
        dst = '{}/{}'.format(folder, file)
        if not os.path.isfile(dst):
            allExist = False
            break
    if not allExist:
        base.logger.info('iRadar sample data not exist. Downloading from server...')
        urllib.request.urlretrieve('https://arrc.ou.edu/iradar/data.zip', 'data.zip')
        with zipfile.ZipFile('data.zip') as zipped:
            for info in zipped.infolist():
                file = '{}/{}'.format(folder, info.filename)
                base.logger.debug('Unzipping file {} ...'.format(info.filename))
                base.logger.debug('-> {} ...'.format(file))
                with open(file, 'wb') as outfile:
                    with zipped.open(info) as zippedfile:
                        outfile.write(zippedfile.read())
            zipfile.close()
        os.remove('data.zip')

def file():
    check(verbose=0)
    return '{}/{}-Z.nc'.format(folder, prefix)

def files():
    check(verbose=0)
    files = ['{}-{}.nc'.format(prefix, x) for x in ['Z', 'V', 'W', 'D', 'P', 'R']]
    return files
