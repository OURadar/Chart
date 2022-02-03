#
# This module is desinged for imgen and imque to reduce repeated codes
#

import os
import re
import textwrap

from .base import logger

def addCommonArguments(parser):
    nameChoices = ['web', 'web90', 'small', 'medium', '720p', '1080p', '1080w', '4k']
    parser.add_argument('-q', action='append', help='specify the spacing of range rings', type=float)
    parser.add_argument('-r', action='append', help='specify the range of the domain', type=float)
    parser.add_argument('-s',
                        action='append',
                        choices=nameChoices,
                        help=textwrap.dedent('specify the size in one of the strings:\n'
                             '    web    - 512 x 584\n'
                             '    small  - 480 x 320\n'
                             '    medium - 960 x 540\n'
                             '    720p   - 1280 x 720\n'
                             '    1080p  - 1920 x 1080 primitive (r = 68)\n'
                             '    1080w  - 1920 x 1080 wide view (r = 200)\n'
                             '    4k     - 3840 x 2160\n'
                             ))
    parser.add_argument('-t', action='append', help='specify the text scale', type=float)
    parser.add_argument('-v', default=0, action='count', help='increase the verbosity level')
    parser.add_argument('-x', action='append', help='specify the x-offset of the camera', type=float)
    parser.add_argument('-y', action='append', help='specify the y-offset of the camera', type=float)
    parser.add_argument('-d', '--dir', action='append', help='specify the destination folder')
    parser.add_argument('-p', '--prefix', action='append', default=None, help='specify the prefix of the files')
    parser.add_argument('-T', '--template',
                        action='append',
                        choices=nameChoices,
                        default=None, help='specify template to use (-s -t -x -y -r -q) (see -s)')
    parser.add_argument('--figsize', action='append', default=None, help='specify the figsize (override -s)')

# Figure properties
def getPropertiesFromArguments(args):
    # Arguments in array form
    def numeric_array_from_string(string, conv=int):
        if re.search(r'[\[(]\d+[+-/x,]\d+[0-9, ]+[\])]', string):
            return ast.literal_eval(string)
        return tuple([conv(x) for x in re.split(r'[+-/x,]', string)])

    def propertiesFromName(size):
        x = 0.0
        y = 0.0
        r = 48.0
        q = 10.0
        t = 1.0
        if size == 'square':
            s = (640, 640)
        elif size == 'small':
            s = (480, 320)
        elif size == 'medium':
            s = (960, 540)
        elif size == '720p':
            s = (1280, 720)
            r = 58.0
            q = 20.0
        elif size == '1080p':
            s = (1920, 1080)
            r = 68.0
            q = 20.0
        elif size == '1080w':
            s = (1920, 1080)
            r = 200.0
            q = 50.0
            t = 1.2
            y = 5
        elif size == '4k':
            s = (3840, 2160)
            r = 68.0
            q = 20.0
        elif size == 'web90':
            s = (512, 582)
            y = 9.0
            r = 72.0
            q = 15.0
        else:
            # Web
            s = (512, 582)
            y = 6.0
        return s, t, x, y, r, q

    # Figure properties
    ss = []
    if args.template:
        count = len(args.template)
        tt = []
        xx = []
        yy = []
        rr = []
        qq = []
        for name in args.template:
            s, t, x, y, r, q = propertiesFromName(name)
            ss.append(s)
            tt.append(t)
            xx.append(x)
            yy.append(y)
            rr.append(r)
            qq.append(q)
    else:
        if args.figsize and args.s:
            logger.info('option --figsize takes precedence over -s')
        if args.figsize:
            if isinstance(args.figsize, list):
                for x in args.figsize:
                    ss.append(numeric_array_from_string(x, int))
        else:
            if not args.s:
                args.s = ['web']
            for name in args.s:
                s, _, _, _, _, _ = propertiesFromName(name)
                ss.append(s)
        tt = args.t if args.t else [1.0]
        xx = args.x if args.x else [0.0]
        yy = args.y if args.y else [0.0]
        rr = args.r if args.r else [58.0]
        qq = args.q if args.q else [10.0]
        count = max([len(x) if isinstance(x, list) else 1 for x in [ss, args.t, args.x, args.y, args.r, args.q]])
        logger.debug('count = {}'.format(count))
        while len(ss) < count:
            ss.append(ss[-1])
        while len(tt) < count:
            tt.append(tt[-1])
        while len(xx) < count:
            xx.append(xx[-1])
        while len(yy) < count:
            yy.append(yy[-1])
        while len(rr) < count:
            rr.append(rr[-1])
        while len(qq) < count:
            qq.append(qq[-1])

    # Output directories and prefixes to the length of count
    dd = []
    if args.dir:
        for d in args.dir:
            dd.append(d.rstrip('/'))
    else:
        dd.append('figs')
    while len(dd) < count:
        dd.append('{}+'.format(dd[-1]))

    # Prefixes
    pp = args.prefix if args.prefix else ['PX']
    while len(pp) < len(ss):
        pp.append(pp[-1])

    # Show the arrays
    logger.debug('args.figsize = {}'.format(args.figsize))
    logger.debug('args.s = {} --> ss = {}'.format(args.s, ss))
    logger.debug('args.t = {} --> tt = {}'.format(args.t, tt))
    logger.debug('args.x = {} --> xx = {}'.format(args.x, xx))
    logger.debug('args.y = {} --> yy = {}'.format(args.y, yy))
    logger.debug('args.r = {} --> rr = {}'.format(args.r, rr))
    logger.debug('args.q = {} --> qq = {}'.format(args.q, qq))
    logger.debug('args.dir = {} --> dd = {}'.format(args.dir, dd))
    logger.debug('args.prefix = {} --> pp = {}'.format(args.prefix, pp))

    # Create the output folders if they don't exist
    for folder in dd:
        folder = os.path.expanduser(folder)
        if not os.path.exists(folder):
            logger.debug('Creating folder {} ...'.format(folder))
            os.makedirs(folder)

    return ss, tt, xx, yy, rr, qq, dd, pp
