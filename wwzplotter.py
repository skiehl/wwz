# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""A class for plotting results of the weighted wavelet z-transform analysis.
"""

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import os
import sys

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# CLASSES
#==============================================================================

class WWZPlotter:
    """A class for plotting WWZ results."""

    #--------------------------------------------------------------------------
    def __init__(self, wwz, tunit=None):
        """A class for plotting WWZ results.

        Parameters
        ----------
        wwz : wwz.WWZ
            A WWZ instance that is used for plotting.

        Returns
        -------
        None.
        """

        self.wwz = wwz
        self.okay = True

        if wwz.wwz is None:
            print('Note: There is no WWZ transform data stored in this WWZ ' \
                  'instance. There will be nothing to plot.')
            self.okay = False

        if wwz.freq is not None:
            # check if frequencies are linearly scaled:
            freq = self.wwz.freq
            df = np.diff(freq)
            self.linear_freq = np.all(np.isclose(df, df.mean()))
            self.fmin = freq.min()
            self.fmax = freq.max()

            # get periods and check if linearly scaled:
            period = 1. / freq
            dp = np.diff(period)
            self.linear_period = np.all(np.isclose(dp, dp.mean()))
            self.pmin = period.min()
            self.pmax = period.max()

            self.n_ybins = freq.size
        else:
            self.okay = False

        if wwz.tau is not None:
            self.tmin = wwz.tau.min()
            self.tmax = wwz.tau.max()
        else:
            self.okay = False

        if self.okay:
            if self.linear_freq:
                self.ymin = self.fmax
                self.ymax = self.fmin
                self.ymin_alt = self.pmax
                self.ymax_alt = self.pmin
                self.ylabel = f'Frequency [1/{tunit}]' \
                        if isinstance(tunit, str) else 'Frequency'
                self.ylabel_alt = f'Period [{tunit}]' \
                        if isinstance(tunit, str) else 'Period'
            elif self.linear_period:
                self.ymin = self.pmin
                self.ymax = self.pmax
                self.ymin_alt = self.fmin
                self.ymax_alt = self.fmax
                self.ylabel = f'Period [{tunit}]' if isinstance(tunit, str) \
                        else 'Period'
                self.ylabel_alt = f'Frequency [1/{tunit}]' \
                        if isinstance(tunit, str) else 'Frequency'
            else:
                self.ymin = 0
                self.ymax = 1
                self.ymin_alt = 1
                self.ymax_alt = 0
                self.ylabel = 'Non-linear scale'
                self.ylabel_alt = 'Non-linear scale'

    #--------------------------------------------------------------------------
    def _select_map(self, select):
        """Helper method to select a map from a WWZ instance.

        Parameters
        ----------
        select : str
            Select either 'wwz' or 'wwa'.

        Raises
        ------
        ValueError
            Raised if 'select' is not one of the allowed options.

        Returns
        -------
        result : numpy.ndarray
            The selected WWZ or WWA array.

        """

        # check that selection is allowed:
        if select.lower() not in ['wwz', 'wwa']:
            raise ValueError(f"'{select}' is not a valid selection.")

        select = select.lower()
        result = eval(f'self.wwz.{select}')

        # check if result map is available:
        if result is None:
            print(f'No {select.upper()} transform available.')

        result = result.transpose()

        return result

    #--------------------------------------------------------------------------
    def plot_map(
            self, select, ax=None, xlabel=None, **kwargs):
        """Plot the resulting map from a WWZ instance.

        Parameters
        ----------
        select : str
            Select either 'wwz' or 'wwa'.
        ax : matplotlib.pyplot.axis, optional
            The axis to plot to. If None is given a new axis is crated. The
            default is None.
        xlabel : str, optional
            The x-axis label. If None is provided no label is placed. The
            default is None.
        kwargs : dict, optional
            Keyword arguments forwarded to the matplotlib.pyplot.imshow()
            function.

        Returns
        -------
        matplotlib.pyplot.axis
            The axis to which the map was plotted.
        matplotlib.image.AxesImage
            The image.
        """

        if not self.okay:
            return None, None

        # select result:
        result = self._select_map(select)
        if result is None:
            return None, None

        # create figure if needed:
        if ax is None:
            __, ax = plt.subplots(1)

        # plot:
        extent = [self.tmin, self.tmax, self.ymin, self.ymax]
        im = ax.imshow(
                result, origin='upper', aspect='auto', extent=extent,
                **kwargs)

        # add labels:
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(self.ylabel)

        return ax, im

    #--------------------------------------------------------------------------
    def plot_map_avg(
            self, select, statistic='mean', ax=None, ylabel=False, **kwargs):
        """Vertically plot an average along the time axis of the transform map.

        Parameters
        ----------
        select : str
            Select either 'wwz' or 'wwa'.
        statistic : str, optional
            Choose either 'mean' or 'median'. The default is 'mean'.
        ax : matplotlib.pyplot.axis, optional
            The axis to plot to. If None is given a new axis is crated. The
            default is None.
        ylabel : bool, optional
            If True a label is added to the y-axis. The default is False.
        **kwargs : dict
            Keyword arguments forwarded to the matplotlib.pyplot.plot()
            function.

        Raises
        ------
        ValueError
            Raised if 'statistic' is not one of the allowed options.

        Returns
        -------
        matplotlib.pyplot.axis
            The axis to which the data was plotted.
        """

        if not self.okay:
            return None

        # select result:
        result = self._select_map(select)
        if result is None:
            return None, None

        # calculate statistic:
        if statistic not in ['mean', 'median']:
            raise ValueError(f"'{statistic}' is not a valid statistic.")
        elif statistic == 'median':
            result_avg = np.median(result, axis=1)
        else:
            result_avg = np.mean(result, axis=1)

        # create figure if needed:
        if ax is None:
            __, ax = plt.subplots(1)

        # plot:
        y = np.linspace(self.ymin, self.ymax, result_avg.size)
        ax.plot(result_avg[::-1], y, **kwargs)

        # add labels:
        if ylabel:
            ax.set_ylabel(self.ylabel)

        ax.set_xlabel(f'{statistic.capitalize()} {select.upper()}')

        return ax

    #--------------------------------------------------------------------------
    def plot_data(
            self, ax=None, errorbars=True, xlabel=None, ylabel=None, **kwargs):
        """Plot the data stored in a WWZ instance.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis, optional
            The axis to plot to. If None is given a new axis is crated. The
            default is None.
        errorbars : bool, optional
            If True errorbars are shown, if uncertainties were stored in the
            WWZ instance. The default is True.
        xlabel : str, optional
            The x-axis description. If None is provided no label is printed.
            The default is None.
        ylabel : str, optional
            The y-axis description. If None is provided no label is printed.
            The default is None.
        **kwargs : dict
            Keyword arguments forwarded to the matplotlib.pyplot.errorbar()
            function.

        Returns
        -------
        matplotlib.pyplot.axis
            The axis to which the data was plotted.
        """

        # check if data is available:
        if self.wwz.t is None:
            print('No data available.')
            return None

        # create figure if needed:
        if ax is None:
            __, ax = plt.subplots(1)

        # plot:
        if errorbars and self.wwz.s_x is not None:
            ax.errorbar(self.wwz.t, self.wwz.x, self.wwz.s_x, **kwargs)
        else:
            ax.plot(self.wwz.t, self.wwz.x, **kwargs)

        # add labels:
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)

        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)

        return ax

    #--------------------------------------------------------------------------
    def add_right_labels(self, ax):
        """Add ticks and labels to the right side of a plot showing the
        alternative unit, i.e. frequency if period is used on the left side and
        vice versa.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis, optional
            The axis to plot to. If None is given a new axis is crated. The
            default is None.

        Returns
        -------
        ax2 : matplotlib.pyplot.axis
            The new axis to which the labels were added.
        """

        ax2 = ax.twinx()
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()

        ax2.set_ylim(self.ymin_alt, self.ymax_alt)

        sys.stderr = open(os.devnull, "w")  # silence stderr to supress warning
        conversion = lambda x: 1/x
        ax2.set_yscale('function', functions=(conversion, conversion))
        sys.stderr = sys.__stderr__  # unsilence stderr
        ax2.yaxis.set_major_locator(LogLocator(subs='all'))
        ax2.set_ylabel(self.ylabel_alt)

        return ax2

    #--------------------------------------------------------------------------
    def plot(self, select, statistic='mean', errorbars=True,
             peaks_quantile=None, xlabel=None, ylabel=None, figsize=None,
             height_ratios=(2, 1), width_ratios=(5, 1), kwargs_map={},
             kwargs_map_avg={}, kwargs_data={}, kwargs_peaks={}):
        """Plot the WWZ map, average, and data.

        Parameters
        ----------
        select : str
            Select either 'wwz' or 'wwa'.
        statistic : str, optional
            Choose either 'mean' or 'median'. The default is 'mean'.
        errorbars : bool, optional
            If True errorbars are shown, if uncertainties were stored in the
            WWZ instance. The default is True.
        peaks_quantile : float, optional
            If not None, a ridge line along the peak position is shown.
            peaks_quantile needs to be a float between 0 and 1. Only peaks in
            the quantile above this threshold are shown. The default is None.
        xlabel : str, optional
            The x-axis description. If None is provided no label is printed.
            The default is None.
        ylabel : str, optional
            The y-axis description. If None is provided no label is printed.
            The default is None.
        figsize : tuple, optional
            Set the figure size. The default is None.
        height_ratios : tuple, optional
            Set the size ratio between the top and bottom panel with two values
            in a tuple. The default is (2, 1).
        width_ratios : tuple, optional
            Set the size ratio between the left and right panel with two values
            in a tuple. The default is (5, 1).
        kwargs_map : dict, optional
            Keyword arguments forwarded to plotting the map. The default is {}.
        kwargs_map_avg : dict, optional
            Keyword arguments forwarded to plotting the map average. The
            default is {}.
        kwargs_data : dict, optional
            Keyword arguments forwarded to plotting the data. The default is
            {}.
        kwargs_peaks : dict, optional
            Keyword arguments forwarded to plotting the peak ridge lines. The
            default is {}.

        Returns
        -------
        ax_map : matplotlib.pyplot.axis
            The map axis.
        ax_map_avg : matplotlib.pyplot.axis
            The map average axis.
        ax_data : matplotlib.pyplot.axis
            The data axis.
        """

        # create figure:
        plt.figure(figsize=figsize)
        grid = gs.GridSpec(
                2, 2, hspace=0, wspace=0, height_ratios=height_ratios,
                width_ratios=width_ratios)
        ax_map = plt.subplot(grid[0,0])
        ax_map_avg = plt.subplot(grid[0,1])
        ax_data = plt.subplot(grid[1,0])

        # plot map:
        self.plot_map(
                select, ax=ax_map, **kwargs_map)

        # plot map average:
        self.plot_map_avg(
                select, statistic=statistic, ax=ax_map_avg, **kwargs_map_avg)
        extend = (self.ymax - self.ymin) / (self.n_ybins - 1) / 2.
        ax_map_avg.set_ylim(self.ymin-extend, self.ymax+extend)

        # plot data:
        self.plot_data(
                ax=ax_data, errorbars=errorbars, xlabel=xlabel, ylabel=ylabel,
                **kwargs_data)
        ax_data.set_xlim(self.tmin, self.tmax)

        # plot peaks:
        if peaks_quantile:
            peak_tau, peak_pos, peak_signal = self.wwz.find_peaks(
                    select, peaks_quantile)
            ax_map.plot(peak_tau, peak_pos, **kwargs_peaks)

        # add right axis labels:
        self.add_right_labels(ax_map_avg)

        # add data axis labels:
        ax_data.set_xlabel(xlabel)
        ax_data.set_ylabel(ylabel)
        plt.setp(ax_map_avg.get_yticklabels(), visible=False)
        plt.setp(ax_map.get_xticklabels(), visible=False)

        return ax_map, ax_map_avg, ax_data

#==============================================================================
