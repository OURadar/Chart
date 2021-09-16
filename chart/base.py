import os
import re
import time
import logging
import tarfile

# Storage home
storageHome = os.path.expanduser('~/Documents/iRadar')
if not os.path.exists(storageHome):
    os.makedirs(storageHome)

# Logger
logging.Formatter.converter = time.gmtime
logHome = os.path.expanduser('~/logs')
logger = logging.getLogger('iRadar')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(message)s', datefmt='%H:%M:%S')

def setLogPrefix(prefix):
    logfile = '{}/{}-{}.log'.format(logHome, prefix, time.strftime('%Y%m%d', time.localtime(time.time())))
    fileHandler = logging.FileHandler(logfile, 'a')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    logger.addHandler(fileHandler)
    return logfile

def getLogfile():
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            return h.baseFilename

def showMessageLevel(level):
    if len(logger.handlers) == 1:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.setLevel(logging.INFO)
        else:
            h.setLevel(level)
    logger.setLevel(min(level, logging.INFO))

def showDebugMessages():
    showMessageLevel(logging.DEBUG)

def showInfoMessages():
    showMessageLevel(logging.INFO)

def shortenPath(path, n):
    parts = path.split('/')
    return '...{}'.format('/'.join(parts[-n:])) if len(parts) > n else path

def extract(archive, path='.'):
    zfile = None
    files = []
    tar = tarfile.open(archive)
    for info in tar.getmembers():
        logger.debug(info.name)
        basename = os.path.basename(info.name)
        if os.path.splitext(basename)[-1] == '.nc':
            tar.extract(info, path=path)
            files.append('{}/{}'.format(path, basename))
            symbols = re.findall(r'(?<=[0-9]-)[A-Za-z]+(?=.nc)', basename)
            if len(symbols) == 0:
                logger.info(archive)
                logger.info(basename)
                continue
            symbol = symbols[0]
            if symbol == 'Z':
                zfile = '{}/{}'.format(path, basename)
    tar.close()
    return zfile, files

# ----------------------------

# if os.path.exists(logHome):
#     setLogPrefix('iradar')
