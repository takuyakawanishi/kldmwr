import matplotlib.pyplot as plt
from matplotlib import lines, rc
#from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import numpy as np
#import pandas as pd
#import sys
#import warnings

rc('text', usetex='False')
rc('mathtext', default='regular')
rc('mathtext', fontset='stix')

WFG = 4.5
FSTL = 7  # Size of Tick Labels
FSTI = 5  # Size of Tick labels of mInor tics
FSAL = 8  # Font size axis labels
FSFA = 7  # Font size figure annotation
LWT = .32  # Line width for thin line.
CSP = 'k'
MKS = 2.7
MSP = 2.7  # Marker Size for Plot
LWP = .32  # Line Width for Plot
LWPT = .8  # Line width for plot (thick)


def color_pallet_cud():
    cud = {
        'red': (255 / 255, 75 / 255, 0),
        'yellow': (255 / 255, 241 / 255, 0),
        'green': (3 / 255, 175 / 255, 122 / 255),
        'blue': (0, 90 / 255, 255 / 255),
        'sky': (77 / 255, 196 / 255, 255 / 255),
        'pink': (255 / 255, 128 / 255, 130 / 255),
        'orange': (246 / 255, 170 / 255, 0),
        'purple': (153 / 255, 0, 153 / 255),
        'brown': (128 / 255, 64 / 255, 0)
    }
    return cud


def color_palette_okabe_ito():
    cud_o_i = {
        'black': (0, 0, 0),
        'orange': (.90, .60, 0),
        'sky blue': (.35, .70, .90),
        'bluish green': (0, .6, .5),
        'yellow': (.95, .90, .25),
        'blue': (0, .45, .7),
        'vermillion': (.8, .4, 0),
        'reddish purple': (.8, .6, .7)
    }
    return cud_o_i


def color_palette_ichihara_et_al():
    cud_i = {
        'black': (0, 0, 0),
        'red': (255 / 255, 59 / 255, 0),
        'blue': (0, 179 / 255, 1),
        'green': (34 / 255, 230 / 255, 92 / 255)
    }
    return cud_i


def color_palette_gs(steps, i):
    gscls = []
    for step in range(steps + 1):
        rel = step / steps
        gscls.append((rel, rel, rel))
    return gscls[i]


def set_figure_dimensions_in_points():
    """ old ones
    ws = 54 # spine width (57, or 54)
    wl = 72 - ws  # width of the label area:
    wyl = wl / 2  # width of yaxis label
    wtl = wl / 2  # width of yaxis tick label
    wfm = 4       # horizontal figure margin
    wsp = 3       # spine pad width, subtract
    hfm = 2       # vertical figure margin (1)
    hs = 36       # spine height 38=57*2/3
    hl = 12       #  height o the label area
    hxl = hl / 2  # height of xaxis label
    htl = hl / 2  # height of xaxis tick label
    hsp = wsp     # spine pad height, subtract
    """
    dimensions = {
        'wfm': 4,  # Width of figure margin
        'hfm': 4,
        'wpg': 4,  # Width of panel gap
        'hpg': 4,
        'ws': 72,
        'wl': 26,
        'wyl': 12,
        'wtl': 14,  # 18 - wpg, 18 makes panel size * 0.25
        'hs': 72,
        'hl': 24,  # 24 * 3 / 4,
        'hxl': 12.0,
        'htl': 12.0,
        'wsp': 0,
        'hsp': 0,
        'wfg': 4,
        'hfg': 4,
    }
    return dimensions


def calc_wt_ht(nhor, nver, syl, sxl, dimensions):
    wfm = dimensions['wfm']
    hfm = dimensions['hfm']
    ws = dimensions['ws']
    wyl = dimensions['wyl']
    wtl = dimensions['wtl']
    wpg = dimensions['wpg']

    hs = dimensions['hs']
    hxl = dimensions['hxl']
    htl = dimensions['htl']
    hpg = dimensions['hpg']

    wt = nhor * ws + 2 * wfm + (nhor - 1) * wpg + wyl * np.sum(syl[:, 0]) + \
         wtl * np.sum(syl[:, 1])
    ht = nver * hs + 2 * hfm + (nver - 1) * hpg + hxl * np.sum(sxl[:, 0]) + \
         htl * np.sum(sxl[:, 1])
    return wt, ht


def create_axes_in_points(nhor, nver, syl, sxl, dimensions):
    syl = np.array(syl)
    sxl = np.array(sxl)
    wfm = dimensions['wfm']
    hfm = dimensions['hfm']
    wpg = dimensions['wpg']
    hpg = dimensions['hpg']

    ws = dimensions['ws']
    wyl = dimensions['wyl']
    wtl = dimensions['wtl']
    wsp = dimensions['wsp']
    hs = dimensions['hs']
    hxl = dimensions['hxl']
    htl = dimensions['htl']
    hsp = dimensions['hsp']
    #
    axs = []
    wt, ht = calc_wt_ht(nhor, nver, syl, sxl, dimensions)

    for ihor in range(nhor):
        for iver in range(nver):
            ax = [
                (wfm + ihor * (wpg + ws) + wsp +
                 wyl * np.sum(syl[:ihor + 1, 0]) +
                 wtl * np.sum(syl[:ihor + 1, 1])),
                #
                ht - hfm - ((iver + 1) * (hs + hpg) - hsp +
                hxl * np.sum(sxl[:iver, 0]) +
                htl * np.sum(sxl[:iver, 1])),
                (ws - wsp),
                (hs - hsp)
            ]
            axs.append(ax)
    return axs


def calc_figure_dimensions(
        n_figures, show_yaxis_label_ticks_g, show_xaxis_label_ticks_g,
        dimensions):
    whfigs = []
    for i in range(n_figures):
        n_panel_horizontal = len(show_yaxis_label_ticks_g[i, :, 0])
        n_panel_vertical = len(show_xaxis_label_ticks_g[i, :, 0])
        #
        # total width, total height of i th figure
        wt, ht = calc_wt_ht(
            n_panel_horizontal, n_panel_vertical,
            show_yaxis_label_ticks_g[i], show_xaxis_label_ticks_g[i],
            dimensions)
        whfigs.append([wt, ht])
    whfigs = np.array(whfigs)
    fig_height = np.sum(whfigs[:, 1]) + (n_figures - 1) * dimensions['hfg']
    fig_width = whfigs[0, 0]
    return fig_width, fig_height, whfigs


def create_figure(fig_width_in_points, fig_height_in_points, enlargement):
    fig_height_inch = fig_height_in_points / 72 * enlargement
    fig_width_inch = fig_width_in_points / 72 * enlargement
    return plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=144)


def create_axes(fig, n_figures, show_yaxis_label_ticks_g,
                show_xaxis_label_ticks_g, dimensions, fig_width, fig_height,
                whfigs):
    axs_in_points = []
    for i in range(n_figures):
        n_panel_horizontal = len(show_yaxis_label_ticks_g[i, :, 0])
        n_panel_vertical = len(show_xaxis_label_ticks_g[i, :, 0])
        axs_in_points.append(
            create_axes_in_points(
                n_panel_horizontal, n_panel_vertical,
                show_yaxis_label_ticks_g[i], show_xaxis_label_ticks_g[i],
                dimensions
            )
        )

    fig_placements = np.zeros((n_figures, 2))
    for i_figure in range(n_figures):
        fig_placements[i_figure, 1] = fig_height - \
                                      np.sum(whfigs[:i_figure, 1]) - dimensions[
                                          'hfg'] * i_figure
        fig_placements[i_figure, 0] = fig_height - \
                                      np.sum(whfigs[:i_figure + 1, 1]) - \
                                      dimensions['hfg'] * i_figure
    global_axs_in_points = np.array(axs_in_points)
    for i_figure in range(n_figures):
        global_axs_in_points[i_figure, :, 1] = \
            global_axs_in_points[i_figure, :, 1] + fig_placements[i_figure, 0]
    gaxs = np.copy(global_axs_in_points)
    gaxs[:, :, 0] = gaxs[:, :, 0] / fig_width
    gaxs[:, :, 1] = gaxs[:, :, 1] / fig_height
    gaxs[:, :, 2] = gaxs[:, :, 2] / fig_width
    gaxs[:, :, 3] = gaxs[:, :, 3] / fig_height
    axss = []
    for i_figure in range(n_figures):
        faxs = []
        for j_panel in range(np.shape(gaxs)[1]):
            # ax = list(gaxs[i_figure, j_panel, :])
            faxs.append(fig.add_axes(gaxs[i_figure, j_panel, :]))
        axss.append(faxs)
    return axss


class TFigure(object):

    def __init__(self, aspect_ratio=1, dimensions=None,
                 syltf=None, sxltf=None, enlargement=1.0,
                 spines_to_pad=None, wpad=None, hpad=None, pad=None
                 ):

        if syltf is None:
            syltf = [[[1, 1]]]
        if sxltf is None:
            sxltf = [[[1, 1]]]
        if dimensions == None:
            self.dimensions = set_figure_dimensions_in_points()
            self.dimensions['hs'] = self.dimensions['ws'] / aspect_ratio
        else:
            self.dimensions = dimensions

        self.show_yaxis_label_ticks_figs = np.array(syltf)
        self.show_xaxis_label_ticks_figs = np.array(sxltf)
        self.enlargement = enlargement

        self.spines_to_pad = spines_to_pad
        if wpad is not None:
            self.dimensions['wsp'] = wpad
        if hpad is not None:
            self.dimensions['hsp'] = hpad
        if pad is not None:
            self.dimensions['wsp'] = pad
            self.dimensions['hsp'] = pad

        self.n_figures = len(self.show_yaxis_label_ticks_figs)
        self.n_panels = []
        self.n_panels_horizontal = []
        self.n_panels_vertical = []

        for i in range(self.n_figures):
            nhp = len(self.show_yaxis_label_ticks_figs[i, :, 0])
            nvp = len(self.show_xaxis_label_ticks_figs[i, :, 0])
            self.n_panels_horizontal.append(nhp)
            self.n_panels_vertical.append(nvp)
            self.n_panels.append(nhp * nvp)

        self.fig_width_in_points, self.fig_height_in_points, self.whfigs = \
            calc_figure_dimensions(
                self.n_figures,
                self.show_yaxis_label_ticks_figs,
                self.show_xaxis_label_ticks_figs,
                self.dimensions
            )

        self.fig_height_mm = self.fig_height_in_points * 25.4 / 72 * \
                             self.enlargement
        self.fig_width_mm = self.fig_width_in_points * 25.4 / 72 * \
                            self.enlargement

        self.fig = create_figure(
            self.fig_width_in_points, self.fig_height_in_points,
            self.enlargement
        )

        self.axs = create_axes(
            self.fig, self.n_figures,
            self.show_yaxis_label_ticks_figs, self.show_xaxis_label_ticks_figs,
            self.dimensions, self.fig_width_in_points,
            self.fig_height_in_points,
            self.whfigs,
        )

        if spines_to_pad is not None:
            self.select_and_pad_spines(spines_to_pad)

        self.set_default_properties()

        self.show_tick_and_tick_labels_according_to_settings()

    def adjust_spines(self, ax, spines):
        for loc, spine in ax.spines.items():
            if loc in spines:
                pad = 0
                if loc == 'bottom':
                    pad = self.dimensions['wsp'] * self.enlargement
                elif loc == 'left':
                    pad = self.dimensions['hsp'] * self.enlargement
                spine.set_position(('outward', pad))
                spine.set_linewidth(LWT)
                spine.set_color(CSP)
            else:
                spine.set_color(None)

    def set_default_properties(self):
        for i in range(self.n_figures):
            for ax in self.axs[i]:
                for spine in ['top', 'right', 'bottom', 'left']:
                    ax.spines[spine].set_linewidth(LWT)
                ax.tick_params(
                    axis='both', which='both', width=LWT,
                    direction='out', labelsize=FSTL
                )
                for tick in ax.get_yticklabels():
                    tick.set_fontname("Arial")
                for tick in ax.get_xticklabels():
                    tick.set_fontname("Arial")

    def select_and_pad_spines(self, spines):
        for i in range(self.n_figures):
            for i_ax, ax in enumerate(self.axs[i]):
                self.adjust_spines(ax, spines)
                ax.tick_params(
                    axis='both', which='both', width=LWT,
                    direction='out', labelsize=FSTL)

                nvp_figi = self.n_panels_vertical[i]
                ihor = np.int(i_ax / nvp_figi)
                ivar = np.mod(i_ax, nvp_figi)
                vis_x = self.show_xaxis_label_ticks_figs[i, ivar, 1]
                vis_y = self.show_yaxis_label_ticks_figs[i, ihor, 1]
                if vis_x == 0:
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                if vis_y == 0:
                    ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)

    def show_tick_and_tick_labels_according_to_settings(self):
        for i in range(self.n_figures):
            for i_ax, ax in enumerate(self.axs[i]):
                nvp_figi = self.n_panels_vertical[i]
                nhp_figi = self.n_panels_horizontal[i]
                ihor = np.int(i_ax / nvp_figi)
                ivar = np.mod(i_ax, nvp_figi)
                ax.xaxis.set_visible(
                    self.show_xaxis_label_ticks_figs[i, ivar, 1]
                )
                ax.yaxis.set_visible(
                    self.show_yaxis_label_ticks_figs[i, ihor, 1]
                )


def main():

    fig = TFigure(
        enlargement=1.2,
        sxltf=[[[0, 0], [1, 1]], [[0, 0], [1, 1]]],
        syltf=[[[1, 1], [0, 0]], [[1, 1], [0, 0]]],
        spines_to_pad=['bottom', 'left'], wpad=6, hpad=6
    )
    print(fig.show_xaxis_label_ticks_figs, fig.show_yaxis_label_ticks_figs)
    print('119 mm wide and not higher than 195 mm.')
    print('figure width: {:.1f}, height: {:1f}'.
          format(fig.fig_width_mm, fig.fig_height_mm))
    print('Number of figures = ', fig.n_figures)
    print('Number of panels  = ', fig.n_panels)
    print('Number of horizontal panels = ', fig.n_panels_horizontal)
    print('Number of vertical panels', fig.n_panels_vertical)
    print(len(fig.axs[0]))

    for axfig in fig.axs:
        for ax in axfig:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            print(ax)
    ws = fig.dimensions['ws']
    wl = fig.dimensions['wl']
    wax = ws - fig.dimensions['wsp']
    wltp = wl + fig.dimensions['wsp']
    el = wltp / wax
    print(ws, wl, wltp, fig.dimensions['wsp'])
    print(el)
    ax = fig.axs[0][0]
    ax.annotate('$a$', xy=(0, 0), xytext=(-el, 0.5),
                textcoords='axes fraction',
                ha='left', va='center')
    # ax.plot([-el, el], [0.5, 0.5], clip_on=False)
    ax.set_xlim(0, 1)
    plt.show()


"""
To Do

* Switch off tick labels according to the indicators.


"""

if __name__ == '__main__':
    main()


"""
def y_scales(nhp, nvp, axs, syl, sxl, dimensions):
    hxl = dimensions['hxl']
    htl = dimensions['htl']
    hfm = dimensions['hfm']
    hs = dimensions['hs']
    hsp = dimensions['hsp']
    wt, ht = calc_wt_ht(nhp, nvp, syl, sxl, dimensions)
    sxl = np.array(sxl)
    x = .5 / nhp
    x2 = x * 0.9
    xsps = x * 0.05
    #
    # Top margin
    #
    axs[0].annotate(
        '', xy=(x, 1), xytext=(x, (ht - hfm) / ht),
        xycoords='figure fraction', textcoords='figure fraction',
        arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                    'linewidth': .5, 'color': 'b', 'alpha': .5},
    )
    axs[0].annotate(
        'hfm:' + str(hfm), xy=(0, 0),
        xytext=(x + xsps, (ht - hfm - 0.1 * (hs - hsp)) / ht),
        textcoords='figure fraction',
        fontsize=6, va='center', c='b', alpha=.5
    )
    for ivp in range(nvp):
        axs[ivp].annotate(
            '', xy=(
                x2, (ht - hfm - ivp * hs - hxl * np.sum(sxl[:ivp, 0]) -
                     htl * np.sum(sxl[:ivp, 1])) / ht),
            xytext=(
                x2, (ht - hfm - (ivp + 1) * hs - hxl * np.sum(sxl[:ivp, 0]) -
                     htl * np.sum(sxl[:ivp, 1])) / ht
            ),
            xycoords='figure fraction', textcoords='figure fraction',
            arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                        'linewidth': .5, 'color': 'b', 'alpha': .5},
        )
        axs[ivp].annotate(
            'hs:' + str(hs), xy=(x2 - xsps, (ht - hfm) / ht),
            xytext=(
                x2 - xsps,
                (ht - hfm - (ivp + .5) * hs - hxl * np.sum(sxl[:ivp, 0]) -
                 htl * np.sum(sxl[:ivp, 1])) / ht),
            xycoords='figure fraction', textcoords='figure fraction',
            fontsize=6, va='center', ha='right', c='b', alpha=.5
        )
        if sxl[ivp, 1]:
            axs[ivp].annotate(
                '', xy=(x2, (ht - hfm - (ivp + 1) * hs -
                             hxl * np.sum(sxl[:ivp, 0]) -
                             htl * np.sum(sxl[:ivp, 1])) / ht),
                xytext=(
                    x2, (ht - hfm - (ivp + 1) * hs -
                         hxl * np.sum(sxl[:ivp, 0]) -
                         htl * np.sum(sxl[:ivp + 1, 1])) / ht),
                xycoords='figure fraction', textcoords='figure fraction',
                arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                            'linewidth': .5, 'color': 'b', 'alpha': .5},
            )
            axs[ivp].annotate(
                'htl:' + str(hsp), xy=(0, 0),
                xytext=(
                    x2 - xsps, (ht - hfm - (ivp + 1) * hs -
                                hxl * np.sum(sxl[:ivp, 0]) -
                                htl * (np.sum(sxl[:ivp + 1, 1]) - 0.5)) / ht),
                textcoords='figure fraction',
                fontsize=6, ha='right', va='center', c='b', alpha=.5
            )
        if sxl[ivp, 0]:
            axs[ivp].annotate(
                '', xy=(x2, (ht - hfm - (ivp + 1) * hs -
                             hxl * np.sum(sxl[:ivp + 1, 0]) -
                             htl * np.sum(sxl[:ivp, 1])) / ht),
                xytext=(
                    x2, (ht - hfm - (ivp + 1) * hs -
                         hxl * np.sum(sxl[:ivp + 1, 0]) -
                         htl * np.sum(sxl[:ivp + 1, 1])) / ht),
                xycoords='figure fraction', textcoords='figure fraction',
                arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                            'linewidth': .5, 'color': 'b', 'alpha': .5},
            )
            axs[ivp].annotate(
                'hxl:' + str(hsp), xy=(0, 0),
                xytext=(
                    x2 - xsps, (ht - hfm - (ivp + 1) * hs -
                                hxl * np.sum(sxl[:ivp + 1, 0]) -
                                htl * (np.sum(sxl[:ivp + 1, 1]) - 0.5)) / ht),
                textcoords='figure fraction',
                fontsize=6, ha='right', va='center', c='b', alpha=.5
            )
        plt.figtext(x + xsps,
                    (ht - hfm - (ivp + 1) * hs -
                     hxl * np.sum(sxl[:ivp, 0]) -
                     htl * np.sum(sxl[:ivp, 1]) + 0.5 * hsp
                     ) / ht,
                    "hsp", fontsize=6, va='center',
                    )
        axs[ivp].annotate(
            '', xy=(x, (ht - hfm - (ivp + 1) * hs -
                        hxl * np.sum(sxl[:ivp, 0]) -
                        htl * np.sum(sxl[:ivp, 1])) / ht),
            xytext=(
                x, (ht - hfm - (ivp + 1) * hs -
                    hxl * np.sum(sxl[:ivp, 0]) -
                    htl * np.sum(sxl[:ivp, 1]) + hsp) / ht),
            xycoords='figure fraction', textcoords='figure fraction',
            arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                        'linewidth': .5, 'color': 'b', 'alpha': .5},
        )
        axs[ivp].annotate(
            'hsp:' + str(hsp), xy=(0, 0),
            xytext=(
                x + xsps, (ht - hfm - (ivp + 1) * hs -
                           hxl * np.sum(sxl[:ivp, 0]) -
                           htl * np.sum(sxl[:ivp, 1]) + 0.5 * hsp) / ht),
            textcoords='figure fraction',
            fontsize=6, va='center', c='b', alpha=.5,
            zorder=1000, clip_on=False,
        )
    return axs


def x_scales(nhp, nvp, axs, syl, sxl, dimensions):
    wyl = dimensions['wyl']
    wtl = dimensions['wtl']
    wfm = dimensions['wfm']
    ws = dimensions['ws']
    wl = dimensions['wl']
    wt, ht = calc_wt_ht(nhp, nvp, syl, sxl, dimensions)
    y = .48
    # y2 = .42
    ysps = .02
    axs[0].annotate(
        '', xy=(0, y), xytext=(wfm / wt, y),
        xycoords='figure fraction', textcoords='figure fraction',
        arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                    'linewidth': .5, 'color': 'b', 'alpha': .5},
    )
    axs[0].annotate(
        'wl:' + str(wl), xy=(0, 0),
        xytext=((wfm + 0.5 * wl) / wt, y - ysps),
        xycoords='figure fraction', textcoords='figure fraction',
        fontsize=6, va='top', ha='center', c='b', alpha=.5
    )
    for ihp in range(nhp):
        ihp_bottom = (ihp + 1) * nvp - 1
        axs[ihp_bottom].annotate(
            '',
            xy=(0 / wt, y),
            xytext=(wfm / wt, y),
            xycoords='figure fraction',
            arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                        'linewidth': LWT, 'color': 'b', 'alpha': .5},
        )
        axs[ihp_bottom].annotate(
            '',
            xy=(
                (ihp * wfm + ihp * ws +
                 wyl * np.sum(syl[:ihp, 0]) +
                 wtl * np.sum(syl[:ihp, 1])) / wt, y
            ),
            xytext=(
                ((ihp + 1) * wfm + ihp * ws +
                 wyl * np.sum(syl[:ihp, 0]) +
                 wtl * np.sum(syl[:ihp, 1])) / wt, y
            ),
            xycoords='figure fraction', textcoords='figure fraction',
            arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                        'linewidth': LWT, 'color': 'b', 'alpha': .5},
        )

        axs[ihp_bottom].annotate(
            '',
            xy=(
                ((ihp + 1) * wfm + ihp * ws +
                 wyl * np.sum(syl[:ihp, 0]) +
                 wtl * np.sum(syl[:ihp, 1])) / wt, y
            ),
            xytext=(
                ((ihp + 1) * wfm + ihp * ws +
                 wyl * np.sum(syl[:ihp + 1, 0]) +
                 wtl * np.sum(syl[:ihp, 1])) / wt, y
            ),
            xycoords='figure fraction', textcoords='figure fraction',
            arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                        'linewidth': LWT, 'color': 'b', 'alpha': .5},
        )
        axs[ihp_bottom].annotate(
            '',
            xy=(
                ((ihp + 1) * wfm + ihp * ws +
                 wyl * np.sum(syl[:ihp + 1, 0]) +
                 wtl * np.sum(syl[:ihp, 1])) / wt, y
            ),
            xytext=(
                ((ihp + 1) * wfm + ihp * ws +
                 wyl * np.sum(syl[:ihp + 1, 0]) +
                 wtl * np.sum(syl[:ihp + 1, 1])) / wt, y
            ),
            xycoords='figure fraction', textcoords='figure fraction',
            arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                        'linewidth': LWT, 'color': 'b', 'alpha': .5},
        )
        axs[ihp_bottom].annotate(
            '',
            xy=(
                ((ihp + 1) * wfm + ihp * ws +
                 wyl * np.sum(syl[:ihp + 1, 0]) +
                 wtl * np.sum(syl[:ihp + 1, 1])) / wt, y
            ),
            xytext=(
                ((ihp + 1) * wfm + (ihp + 1) * ws +
                 wyl * np.sum(syl[:ihp + 1, 0]) +
                 wtl * np.sum(syl[:ihp + 1, 1])) / wt, y
            ),
            xycoords='figure fraction', textcoords='figure fraction',
            arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0,
                        'linewidth': LWT, 'color': 'b', 'alpha': .5},
        )
    return axs

def draw_dimensions(show, axs, n_panel_horizontal, n_panel_vertical,
                    show_yaxis_label_ticks_g, show_xaxis_label_ticks_g,
                    dimensions, i):
    if show:
        axs = y_scales(
            n_panel_horizontal, n_panel_vertical, axs,
            show_yaxis_label_ticks_g[i], show_xaxis_label_ticks_g[i],
            dimensions
        )
        axs = x_scales(
            n_panel_horizontal, n_panel_vertical, axs,
            show_yaxis_label_ticks_g[i], show_xaxis_label_ticks_g[i],
            dimensions
        )
    return axs


"""