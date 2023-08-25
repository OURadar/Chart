from . import font
from . import colormap
from . import base
import PIL
import numpy as np
import matplotlib.pyplot
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('agg')


if matplotlib.__version__ < '3.0.2':
    base.logger.warning('This module requires matplotlib 3.0.2 or newer')

matplotlib.pyplot.ioff()

bgColor = (0, 0, 0)


class Image:
    def padgrids(self, e, a, r):
        delta_a = np.diff(a)
        delta_a[delta_a > +np.pi] -= 2.0 * np.pi
        delta_a[delta_a < -np.pi] += 2.0 * np.pi
        delta_a = delta_a.mean()

        delta_e = np.diff(e)
        delta_e[delta_e > +np.pi] -= 2.0 * np.pi
        delta_e[delta_e < -np.pi] += 2.0 * np.pi
        delta_e = delta_e.mean()

        delta_r = np.diff(r).mean()

        # Pad on extra element since they are the bounds
        r = np.append(r, r[-1] + delta_r)
        a = np.append(a, a[-1] + delta_a)
        e = np.append(e, e[-1] + delta_e)
        return e, a, r

        # rr, aa = np.meshgrid(r, a)
    def updateQuads(self, e, a, r):
        tmp_dict = {'rlat':rlat,'rlon':rlon,'radar_elev':radar_elev,\
            'range':r, 'az':a,  'theta':e}
        glon, glat, gh, gx, gy, gs = radar_navigation(tmp_dict)
        if self.scantype == 'PPI':
            self.xx = gx
            self.yy = gy
        elif self.scantype == 'RHI':
            self.xx = gs
            self.yy = gh
        if self.dmesh:
            self.dmesh.remove()
        self.dmesh = None

    def __init__(self, a=None, r=None, values=None, x=0.0, y=0.0, t=1.0, maxrange=50.0,
                 style='Z', scantype = 'PPI', symbol=None, title=None, dpi=72, figsize=(1280, 720), overlay=None, pcolorfast=True):
        self._dpi = dpi
        self._figsize = figsize
        self.featureScale = t
        self.fontproperties = font.Properties(scale=t)
        self.pcolorfast = pcolorfast
        self.scantype = scantype

        # Use a separate set of context properties
        context_properties = {
            'font.family': 'serif',
            'font.serif': ['Arial', 'DejaVu Serif'],
            'axes.edgecolor': 'white',
            'axes.facecolor': 'black',
            'axes.labelcolor': 'white',
            'axes.linewidth': 1.0 * self.featureScale,
            'axes.titlepad': 8.0 * self.featureScale,
            'grid.color': 'white',
            'hatch.color': 'white',
            'text.color': 'white',
            'xtick.color': 'white',
            'xtick.direction': 'in',
            'xtick.major.pad': 9.0 * self.featureScale,
            'xtick.major.size': 4.0 * self.featureScale,
            'xtick.major.width': 1.0 * self.featureScale,
            'ytick.color': 'white',
            'ytick.direction': 'in',
            'ytick.major.pad': 7.5 * self.featureScale,
        }
        with matplotlib.pyplot.rc_context(context_properties):
           # Create a new figure
            # Axis width & height in points
            w, h = self._figsize[0], self._figsize[1]
            if w > h:
                xmin = -maxrange * w / h + x
                xmax = +maxrange * w / h + x
                ymin = -maxrange + y
                ymax = +maxrange + y
            else:
                xmin = -maxrange + x
                xmax = +maxrange + x
                ymin = -maxrange * h / w + y
                ymax = +maxrange * h / w + y
            # Figure size in inches
            figsize = np.array([w, h]) / self._dpi
            self.fig_dat = matplotlib.pyplot.figure(
                figsize=figsize, dpi=self._dpi, frameon=False)
            self.dat = self.fig_dat.add_axes([0.0, 0.0, 1.0, 1.0])
            self.dat.set_xlim((xmin, xmax))
            self.dat.set_ylim((ymin, ymax))
            self.dat.xaxis.set_visible(False)
            self.dat.yaxis.set_visible(False)
            self.dat.spines['top'].set_visible(False)
            self.dat.spines['right'].set_visible(False)
            self.dat.spines['bottom'].set_visible(False)
            self.dat.spines['left'].set_visible(False)
            self.dat.patch.set_facecolor(bgColor)
            self.fig_dat.canvas.draw()
            base.logger.debug(
                'extent = [{:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(xmin, xmax, ymin, ymax))
            base.logger.debug('fig_dat size = {}'.format(
                self.fig_dat.get_size_inches() * self._dpi))

            # Create the overlay figure
            self.fig_map = matplotlib.pyplot.figure(
                figsize=figsize, dpi=self._dpi, frameon=False)
            self.map = self.fig_map.add_axes([0.0, 0.0, 1.0, 1.0])
            self.map.set_xlim((xmin, xmax))
            self.map.set_ylim((ymin, ymax))
            self.map.xaxis.set_visible(False)
            self.map.yaxis.set_visible(False)
            self.map.spines['top'].set_visible(False)
            self.map.spines['right'].set_visible(False)
            self.map.spines['bottom'].set_visible(False)
            self.map.spines['left'].set_visible(False)
            self.map.patch.set_facecolor(bgColor)
            self.map.patch.set_alpha(0.0)
            self.fig_map.canvas.draw()
            base.logger.debug('fig_map size = {}'.format(
                self.fig_map.get_size_inches() * self._dpi))
            if overlay:
                self.overlay = overlay
                self.overlay.draw(self.map)
            self.fig_map.canvas.draw()
            base.logger.debug('fig_map size = {}'.format(
                self.fig_map.get_size_inches() * self._dpi))

            # Create the top-bar figure
            bw = w
            bh = int(70.0 * self.featureScale)
            self.bw = bw
            self.bh = bh
            figsize = np.array([bw, bh]) / self._dpi
            self.fig_bar = matplotlib.pyplot.figure(
                figsize=figsize, dpi=self._dpi, frameon=False)
            self.bar = self.fig_bar.add_axes([0.0, 0.0, 1.0, 1.0])
            self.bar.set_xlim((0.0, bw))
            self.bar.set_ylim((0.0, bh))
            self.bar.xaxis.set_visible(False)
            self.bar.yaxis.set_visible(False)
            self.bar.spines['top'].set_visible(False)
            self.bar.spines['right'].set_visible(False)
            self.bar.spines['bottom'].set_visible(False)
            self.bar.spines['left'].set_visible(False)
            self.dat.patch.set_facecolor(bgColor)
            self.bar.patch.set_alpha(0.0)
            self.bar.add_line(matplotlib.lines.Line2D([0, bw], [1.5, 1.5],
                                                      color='white', alpha=0.5, linewidth=self.featureScale))
            self.fig_bar.canvas.draw()
            base.logger.debug('fig_bar size = {}'.format(
                self.fig_bar.get_size_inches() * self._dpi))

            # Paint the gradient
            z = colormap.fleximap(20, [0.0, 0.50, 0.51, 1.0],
                                  [[0.33, 0.60, 0.69, 0.70],
                                   [0.12, 0.39, 0.48, 0.70],
                                   [0.00, 0.27, 0.36, 0.70],
                                   [0.00, 0.27, 0.36, 0.70]])
            z = z.reshape((20, 1, 4))
            self.bar.imshow(z, extent=(0.0, bw, 0.0, bh))

            # Big symbol
            self.symbol_text = self.bar.text(0.4 * bh, 0.5 * bh, '-',
                                             fontproperties=self.fontproperties.symbol,
                                             verticalalignment='center_baseline')

            # Colormaps
            def makecolormap(colors):
                return matplotlib.colors.LinearSegmentedColormap.from_list('colors', colors[:, :3], N=len(colors))
            self.colormaps = {}
            self.colormaps['Z'] = makecolormap(colormap.zmap())
            self.colormaps['V'] = makecolormap(colormap.vmap())
            self.colormaps['W'] = makecolormap(colormap.wmap())
            self.colormaps['D'] = makecolormap(colormap.dmap())
            self.colormaps['P'] = makecolormap(colormap.pmap())
            self.colormaps['R'] = makecolormap(colormap.rmap())
            self.colormaps['K'] = makecolormap(colormap.kmap())
            self.colormaps['i'] = makecolormap(colormap.imap())
            self.colormaps['x'] = makecolormap(colormap.zmapx())
            self.colormaps['Zc'] = makecolormap(colormap.zmap())
            self.colormaps['Dc'] = makecolormap(colormap.dmap())
            self.colormaps['Vn'] = makecolormap(colormap.rgmap())
            self.colormaps['i2'] = makecolormap(colormap.i2map())

            # Colorbar
            ct = self.fontproperties.title.get_size_in_points()
            # Colorbar height in points
            ch = np.round(16.0 * self.featureScale)
            # Colorbar padding height in points
            pw = np.round(20.0 * self.featureScale)
            # Colorbar padding width in points
            ph = np.round(12.0 * self.featureScale)
            # Colorbar width in points
            cw = int(np.floor(self.featureScale * 2.0) * 256)
            # Budget 100 points for the big symbol
            while (cw > w - self.featureScale * 100.0):
                if w < 640:
                    cw -= 128
                else:
                    cw -= 256
            if pcolorfast:
                rect = [bw - cw - pw, bh - ch - ph - ct, cw, ch]
            else:
                rect = [bw - cw - pw, bh - ch - ph - ct, cw, ch]
            rect = [n / d for n, d in zip(rect, [bw, bh, bw, bh])]
            self.cax = self.fig_bar.add_axes(rect, facecolor=None)
            self.cax.yaxis.set_visible(False)
            self.cax.set_alpha(0.0)
            self.cw = cw

            # Compute number of pixels per shade, then the pixel adjustments
            s = cw / 256.0
            self.cs = (s - np.ceil(self.featureScale)) / np.ceil(s)
            base.logger.debug('cw = {}   self.cs = {:.4f}'.format(cw, self.cs))

            self.dmesh = None
            self.cmesh = None

        # When Chart is initialized as chart.Chart(z)
        if a is not None and r is None and values is None:
            values = a
            a = None
            r = None
        # Initialize arrays of coordinates
        if values is None:
            if a is None:
                a = np.arange(360, dtype=float) * np.pi / 180.0
            if r is None:
                r = np.arange(1000, dtype=float) * 60.0
            self.updateQuads(a, r)
            self.values = None
        else:
            if a is None:
                a = np.arange(
                    values.shape[0], dtype=float) * 2.0 * np.pi / values.shape[0]
            if r is None:
                r = np.arange(values.shape[1], dtype=float) * 0.06
            self.updateQuads(a, r)
            self.set_data(values, style=style, symbol=symbol, title=title)
        self.fig_bar.canvas.draw()

    # Data is set with values and a set of built-in styles
    #
    # Convention:
    # Data value is in free form, float type
    # Colorbar is always shown in index, assuming 256 shades
    # XTicks is specified in the integer form in, [0, 256] (256 colors, 257 boundaries)
    # XTickLabels can be any text that matplotlib supports
    #
    # Non-obvious parameter
    # - vlim corresponds to the lower and upper limits of the values
    # - cticks corresponds to the limits of the colorbar bar, which is always in [0...255]
    # - cticklabels corresponds to the labels set in cticks
    def set_data(self, values, a=None, r=None, style='S', symbol=None, title=None,
                 vlim=None, cticks=None, cticklabels=None):
        if a is not None and r is not None:
            self.updateQuads(a, r)
        mask = np.isfinite(values)
        base.logger.debug('fig_dat size = {}'.format(
            self.fig_dat.get_size_inches() * self._dpi))
        base.logger.debug('fig_map size = {}'.format(
            self.fig_map.get_size_inches() * self._dpi))

        # Pick a colormap, vmin, vmax, ticklabels, titlestring, etc. based on style
        if style == 'K':
            # KDP is not finalized yet
            slim = (-10.0, +10.0)
            sticklabels = np.arange(-10, 10, 2)
            sticks = sticklabels * 128.0 / 10.0 + 128.0
            titlestring = 'KDP (degrees / km)'
        elif style == 'R':
            # Special case, values are mapped to indices
            slim = (0.0, 256.0)
            values = np.copy(values)
            values[mask] = rho2ind(values[mask])
            sticklabels = np.array([0.73, 0.83, 0.93, 0.96, 0.99, 1.02, 1.05])
            sticks = rho2ind(sticklabels)
            titlestring = 'RhoHV (unitless)'
        elif style == 'P':
            d = 360.0 / 256.0
            slim = (-180.0, +180.0)
            sticklabels = np.arange(-135, 151, 45)
            sticks = sticklabels * 128.0 / 180.0 + 128.0
            titlestring = 'PhiDP (degrees)'
        elif style == 'D' or style == 'Dc':
            slim = (-10.0, +15.6)
            sticklabels = np.arange(-9, 15, 3)
            sticks = sticklabels * 10.0 + 100.0
            titlestring = 'ZDR (dB)'
        elif style == 'W':
            # I realize there is an offset of 1 but okay
            slim = (0, 12.80)
            sticklabels = np.arange(0, 13, 2)
            sticks = sticklabels * 20.0
            titlestring = 'Width (m/s)'
        elif style == 'V':
            # slim = (-32.0, +32.0)
            # sticklabels = np.arange(-24, 25, 8)
            # sticks = sticklabels * 128.0 / 32.0 + 128.0
            slim = (-64, +64.0)
            sticklabels = np.arange(-60, 61, 15)
            sticks = sticklabels * 128.0 / 64.0 + 128.0
            titlestring = 'Velocity (m/s)'
        elif style == 'Z' or style == 'Zc':
            # colors = colormap.zmap()
            slim = (-32.0, +96.0)
            sticklabels = np.arange(-25, 81, 15)
            sticks = sticklabels * 2.0 + 64.0
            titlestring = 'Reflectivity (dBZ)'
        elif style == 'test':
            style = 'x'
            slim = tuple(x * 0.5 - 32.0 for x in (0, 256))
            sticks = np.append(0, np.arange(4, 256, 10))
            sticklabels = None
            titlestring = 'Test'
        elif style == 'i' or style == 'I':
            style = 'i'
            slim = (-0.5, 7.5)
            sticks = np.arange(0, 256, 32) + 16
            sticklabels = None
            titlestring = 'Index'
        elif style == 'i2':
            slim = (-0.5, 1.5)
            sticks = np.arange(0, 256, 128) + 64
            sticklabels = None
            titlestring = 'Index'
        else:
            slim = tuple(x * 0.5 - 32.0 for x in (0, 256))
            sticks = np.array([0, 14, 44, 74, 104, 134, 164, 194, 255])
            sticklabels = None
            titlestring = 'Data'
        # Colormap
        cmap = self.colormaps[style]
        # If the user did not supply vlim, use the limits specified in the style
        if vlim is None and slim is not None:
            vlim = slim
        # Similarly, if the user did not supply the ticks, use the ticks specified in the style
        if cticks is None and sticks is not None:
            cticks = sticks
        if cticklabels is None and sticklabels is not None:
            cticklabels = sticklabels
        # Keep a copy of the values
        self.values = np.ma.masked_where(~mask, values)
        # Paint the main area (twice if this is the first run)
        if self.dmesh is None:
            if self.pcolorfast:
                self.dmesh = self.dat.pcolorfast(self.xx, self.yy, self.values)
            else:
                self.dmesh = self.dat.pcolormesh(self.xx, self.yy, self.values)
        if self.cmesh is None:
            if self.pcolorfast:
                self.cmesh = self.cax.pcolorfast(256.0 / self.cw * np.arange(self.cw + 1), np.arange(3),
                                                 256.0 / self.cw *
                                                 np.arange(self.cw).reshape(
                                                     (1, self.cw)).repeat(2, axis=0) + 0.01,
                                                 vmin=0, vmax=256)
            else:
                self.cmesh = self.cax.pcolormesh(256.0 / self.cw * np.arange(self.cw + 1), np.arange(3),
                                                 256.0 / self.cw *
                                                 np.arange(self.cw).reshape(
                                                     (1, self.cw)).repeat(2, axis=0) + 0.01,
                                                 vmin=0, vmax=256)
        self.dmesh.set_array(self.values.ravel())
        self.dmesh.set_clim(vlim)
        self.dmesh.set_cmap(cmap)
        self.cmesh.set_cmap(cmap)
        # Title, ticks, limits, etc.
        # base.logger.debug(cticks)
        # base.logger.debug(cticklabels)
        # if self.featureScale > 1.0:
        #     self.cax.set_xticks(cticks + self.cs)
        # else:
        self.cax.set_xticks(cticks)
        if cticklabels is None:
            self.cax.set_xticklabels(cticks)
        else:
            self.cax.set_xticklabels(cticklabels)
        for tick in self.cax.xaxis.get_ticklabels():
            tick.set_fontproperties(self.fontproperties.tick)
        if title:
            titlestring = title
        self.cax.set_title(
            titlestring, fontproperties=self.fontproperties.title, pad=8.0 * self.featureScale)
        if symbol:
            self.symbol_text.set_text(symbol)
        else:
            self.symbol_text.set_text(style)
        self.fig_dat.canvas.draw()
        self.fig_bar.canvas.draw()
        # A workaround when im_map has different size
        target_size = [int(x)
                       for x in self.fig_map.get_size_inches() * self._dpi]
        raster_size = [
            int(x) for x in self.fig_map.canvas.renderer.get_canvas_width_height()]
        if not target_size == raster_size:
            self.fig_map.canvas.draw()
            updated_size = [
                int(x) for x in self.fig_map.canvas.renderer.get_canvas_width_height()]
            base.logger.debug(
                'Rasterized map axes. {} / {} -> {}'.format(target_size, raster_size, updated_size))

    def pixel_buffer(self):
        def blend(src, dst, dst_is_opaque=True):
            if dst_is_opaque:
                a_src = np.expand_dims(src[:, :, 3] / 255.0, axis=2)
                # c_out = src[:, :, :3] * a_src + dst[:, :, :3] * (1.0 - a_src)
                c_out = dst[:, :, :3] + (src[:, :, :3] - dst[:, :, :3]) * a_src
                return c_out
            else:
                a_src = np.expand_dims(src[:, :, 3] / 255.0, axis=2)
                a_dst = np.expand_dims(dst[:, :, 3] / 255.0, axis=2)
                a_out = a_src + a_dst * (1.0 - a_src)
                # c_out = (src[:, :, :3] * a_src + dst[:, :, :3] * a_dst * (1.0 - a_src)) / a_out
                a_dst_x_dst = a_dst * dst[:, :, :3]
                c_out = ((src[:, :, :3] - a_dst_x_dst)
                         * a_src + a_dst_x_dst) / a_out
                a_out *= 255.0
                return np.concatenate((c_out, a_out), axis=2)
        if self.values is None:
            values = np.empty(self.xx.shape)
            values[:] = np.nan
            self.set_data(values[:-1, :-1], style='test', title='title')
        # Blend all layers
        im_bar = np.array(
            self.fig_bar.canvas.renderer._renderer, dtype=float)
        im_map = np.array(
            self.fig_map.canvas.renderer._renderer, dtype=float)
        im_dat = np.array(
            self.fig_dat.canvas.renderer._renderer, dtype=float)
        x = blend(im_map, im_dat)
        x[:self.bh, :, :] = blend(im_bar[:self.bh, :, :], x[:self.bh, :, :])
        return x

    def image(self):
        return PIL.Image.fromarray(np.array(self.pixel_buffer(), dtype=np.uint8), 'RGB')

    def savefig(self, filename):
        image = self.image()
        image.save(filename)

    def close(self):
        matplotlib.pyplot.close(self.fig_dat)
        matplotlib.pyplot.close(self.fig_map)
        matplotlib.pyplot.close(self.fig_bar)


def rho2ind(values):
    m3 = values > 0.93
    m2 = np.logical_and(values > 0.7, ~m3)
    index = values * 52.8751
    index[m2] = values[m2] * 300.0 - 173.0
    index[m3] = values[m3] * 1000.0 - 824.0
    return np.round(index)


def image_from_pixel_buffer(buff):
    return PIL.Image.fromarray(np.array(buff, dtype=np.uint8), 'RGB')

def radar_navigation(radar_dict):
    radar_lon=radar_dict['rlon']
    radar_lat=radar_dict['rlat']
    radar_theta=radar_dict['theta']
    radar_az=radar_dict['az']
    rng, az = np.meshgrid(radar_dict['range'], radar_az)
    rng, ele = np.meshgrid(radar_dict['range'], radar_theta)
    # theta_e = ele * np.pi / 180.0       # elevation angle in radians.
    # theta_a = az * np.pi / 180.0        # azimuth angle in radians.
    Re = 6371.0 * 1000.0 * 4.0 / 3.0    # effective radius of earth in meters.
    r = rng * 1000.0                    # distances to gates in meters.

    z = (r ** 2 + Re ** 2 + 2.0 * r * Re * np.sin(theta_e)) ** 0.5 - Re
    z = z + radar_dict['radar_elev']
    s = Re * np.arcsin(r * np.cos(theta_e) / (Re + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    Re = 6371.0 * 1000.0                # radius of earth in meters.
    c = np.sqrt(x*x + y*y) / Re
    phi_0 = radar_lat * np.pi / 180
    azi = np.arctan2(y, x)  # from east to north

    lat = np.arcsin(np.cos(c) * np.sin(phi_0) +
                    np.sin(azi) * np.sin(c) * np.cos(phi_0)) * 180 / np.pi
    lon = (np.arctan2(np.cos(azi) * np.sin(c), np.cos(c) * np.cos(phi_0) -
           np.sin(azi) * np.sin(c) * np.sin(phi_0)) * 180 /
            np.pi + radar_lon)
    lon = np.fmod(lon + 180, 360) - 180
    height = z
    # height, tmp = np.meshgrid(z, radar_az)
    return lon, lat, height, x, y, s
