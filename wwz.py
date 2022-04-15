# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""A class for weighted wavelet z-transform analysis.
"""

from datetime import datetime
import numpy as np
import pickle
from scipy.interpolate import interp1d, splrep, splev
import sys

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann", "Walther Max-Moerbeck", "Oliver King"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# CLASSES
#==============================================================================

class WWZ:
    """A class for weighted wavelet z-transform analysis."""

    #--------------------------------------------------------------------------
    def __init__(self, time=None, flux=None, flux_err=0):
        """A class for weighted wavelet z-transform analysis.

        Parameters
        ----------
        time : array-like, optional
            Time stamps of the input data. The default is None.
        flux : array-like, optional
            Flux of the input data. The default is None.
        flux_err : array-like or float, optional
            Flux uncertainty of the input data. The default is 0.

        Returns
        -------
        None.

        Notes
        -----
        Setting the data during the instance initialization is optional. Data
        can be provided with the set_data() method.
        """

        print('WWZ instance created.')

        if time is not None and flux is not None:
            self.set_data(time, flux, flux_err=flux_err)
        else:
            self._reset()

    #--------------------------------------------------------------------------
    def _reset(self):
        """Reset all class attributes to None.

        Returns
        -------
        None.
        """

        # data time, flux, and flux error:
        self.t = None
        self.x = None
        self.s_x = None

        # number of data points:
        self.n = 0

        # attributes filled by set_freq() and set_tau():
        self.linear_period = False
        self.freq = None
        self.tau = None
        self.len0 = None
        self.len1 = None

        # class attributes filled by transform method:
        self.snr_weights = None
        self.c = None
        self.v_x = None
        self.v_y = None
        self.n_eff = None
        self.wwz = None
        self.wwa = None
        self.wwz_sig = None
        self.wwa_sig = None
        self.wwz_pval = None
        self.wwa_pval = None
        self.n_sig = None

    #--------------------------------------------------------------------------
    def set_data(self, time, flux, flux_err=0.0, verbose=True):
        """Provide data for the analysis.

        Parameters
        ----------
        time : array-like, optional
            Time stamps of the input data. The default is None.
        flux : array-like, optional
            Flux of the input data. The default is None.
        flux_err : array-like or float, optional
            Flux uncertainty of the input data. The default is 0.
        verbose : bool, optional
            Turns printing of information on or off. The default is True.

        Returns
        -------
        None.

        Notes
        -----
        When new data is provided all class attributes that may contain the
        results of an earlier analysis will be reset to None.
        """

        self._reset()

        # data time, flux, and flux error:
        self.t = np.asarray(time)
        self.x = np.asarray(flux)
        self.s_x = np.asarray(flux_err)

        # number of data points:
        self.n = len(time)

        if verbose:
            print('Data stored.')

    #--------------------------------------------------------------------------
    def get_period_lims(self, p_min_factor=4., p_max_factor=5.):
        """Get suggested limits for the periods.

        Parameters
        ----------
        p_min_factor : float, optional
            If p_min is None, p_min is calculated as the median difference
            between time steps multiplied with this factor.
        p_max_factor : float, optional
            If p_max is None, p_max is calculated as the total time of the data
            devided by this factor.

        Returns
        -------
        p_min : float
            Suggested minimum period.
        p_max : float
            Suggested maximum period.
        """

        p_min = np.round(np.median(np.diff(self.t)) * p_min_factor)
        p_max = np.round((self.t[-1] - self.t[0]) / p_max_factor)

        return p_min, p_max

    #--------------------------------------------------------------------------
    def get_freq(
            self, p_min=None, p_max=None, n_bins=100, diff=None,
            linear_period=False, p_min_factor=4., p_max_factor=5.):
        """Create a range of frequencies for the analysis.

        Parameters
        ----------
        p_min : float, optional
            Minimum period. If provided, this shorest period will define the
            highest frequency. If None, the shortest period will be determined
            by the median of the time difference between data points. The
            default is None.
        p_max : float, optional
            Maximum period. If provided, this largest period will define the
            lowest frequency. If None, the largest period will be determined
            by the total duration of the data. The default is None.
        n_bins : int, optional
            Number of frequencies. The default is 100.
        diff : float, optional
            Sets the difference between frequency bins. The number of bins will
            be calculated accordingly. If linear_period=True, this sets the
            difference between periods. If diff is set, this overwrites any
            input to n_bins. The default is None.
        linear_period : bool, optional
            If False, the frequencies will be distributed in equal steps. If
            True, the frequencies are chosen such that the corresponding
            periods are distributed in equal steps. The default is False.
        p_min_factor : float, optional
            If p_min is None, p_min is calculated as the median difference
            between time steps multiplied with this factor.
        p_max_factor : float, optional
            If p_max is None, p_max is calculated as the total time of the data
            devided by this factor.

        Raises
        ------
        ValueError
            Error is raise when neither n_bins nor diff is provided.

        Returns
        -------
        freq : numpy.ndarray
            One dimensional array of frequencies.
        """

        p_min_suggested, p_max_suggested =  self.get_period_lims(
                p_min_factor=p_min_factor, p_max_factor=p_max_factor)

        if p_min is None:
            p_min = p_min_suggested
        elif p_min < p_min_suggested:
            print('WARNING: p_min is shorter than {0} '.format(p_min_factor),
                  'times the median time step of the data.\np_min should ',
                  'not be shorter than {0}.\n'.format(p_min_suggested))

        if p_max is None:
            p_max = p_max_suggested
        elif p_max > p_max_suggested:
            print('WARNING: p_max is larger than the maximum time range',
                  'of the data devided by {0}.\np_max '.format(p_max_factor),
                  'should not be larger than {0}.\n'.format(p_max_suggested))

        freq_min = 1. / p_max
        freq_max = 1. / p_min

        if linear_period:
            if diff is not None:
                period = np.arange(p_min, p_max, diff)

            elif isinstance(n_bins, int):
                period = np.linspace(p_min, p_max, n_bins)
                diff = period[1] - period[0]

            else:
                raise ValueError("Either n_bins or diff needs to be set.")

            freq = 1. / period
            freq = freq[::-1]

        else:
            if diff is not None:
                freq = np.arange(freq_min, freq_max, diff)

            elif isinstance(n_bins, int):
                freq = np.linspace(freq_min, freq_max, n_bins)
                diff = freq[1] - freq[0]

            else:
                raise ValueError("Either n_bins or diff needs to be set.")

        print('Linear range of frequencies created with')
        print('Shortest period:       {0:10.2e}'.format(p_min))
        print('Longest period:        {0:10.2e}'.format(p_max))
        if linear_period:
            print('Period interval:       {0:10.2e}'.format(diff))
        else:
            print('Period interval:       non-linear')
        print('Lowest frequency:      {0:10.2e}'.format(freq_min))
        print('Highest frequency:     {0:10.2e}'.format(freq_max))
        if linear_period:
            print('Frequency interval:    non-linear')
        else:
            print('Frequency interval:    {0:10.2e}'.format(diff))
        print('Number of frequencies: {0:10d}\n'.format(freq.size))

        return freq

    #--------------------------------------------------------------------------
    def get_tau(self, t_min=None, t_max=None, n_div=8, n_bins=None, dtau=None):
        """Create a range of taus (time points) for the analysis.

        Parameters
        ----------
        t_min : float, optional
            Earliest time point. If not provided, will be the first data point.
            The default is None.
        t_max : float, optional
            Latest time point. If not provided, will be the last data point.
            The default is None.
        n_div : int, optional
            The total time of the data will be devided by n_div. The result is
            converted to the nearest integer. This sets the number of time
            bins. A larger n_div will result in fewer time bins, and vice
            versa. The default is 8.
        n_bins : int, optional
            Number of time bins. If n_bins is set, this overwrites any input to
            n_div. The default is None.
        dtau : float, optional
            Sets the difference between time bins. The number of bins will
            be calculated accordingly. If diff is set, this overwrites any
            input to n_bins and n_div. The default is None.

        Raises
        ------
        ValueError
            Error is raise when neither n_bins nor diff is provided.

        Returns
        -------
        freq : numpy.ndarray
            One dimensional array of frequencies.
        """

        t_min = self.t[0]
        t_max = self.t[-1]

        if dtau is not None:
            tau = np.arange(t_min, t_max, dtau)
            n_bins = tau.size

        elif isinstance(n_bins, int):
            tau = np.linspace(t_min, t_max, n_bins)
            dtau = tau[1] - tau[0]

        elif isinstance(n_div, int):
            n_bins = int((t_max - t_min) / n_div)
            tau = np.linspace(t_min, t_max, n_bins)
            dtau = tau[1] - tau[0]

        else:
            raise ValueError("Either n_div, n_bins, or dtau needs to be set.")

        print('Linear range of tau (time points) created with')
        print('Earliest time:         {0:10.1f}'.format(t_min))
        print('Latest time:           {0:10.1f}'.format(t_max))
        print('Time interval:         {0:10.1f}'.format(dtau))
        print('Points in time:        {0:10d}\n'.format(n_bins))

        return tau

    #--------------------------------------------------------------------------
    def set_freq(self, freq, verbose=True):
        """Set the frequencies for the analysis.

        Parameters
        ----------
        freq : array-like
            A range of frequencies at which the WWZ transform will be
            calculated.
        verbose : bool, optional
            Turns printing of information on or off. The default is True.

        Returns
        -------
        None.

        Notes
        -----
        It is recommended to use the get_freq() method to create the array of
        frequencies.
        """

        self.freq = np.asarray(freq)
        self.len1 = freq.size

        # check if frequencies are linear in period space:
        dperiod = np.diff(1. / freq)
        self.linear_period = np.all(np.isclose(dperiod, dperiod[0]))

        if verbose:
            print('Frequencies set.')

    #--------------------------------------------------------------------------
    def set_tau(self, tau, verbose=True):
        """Set the taus (time points) for the analysis.

        Parameters
        ----------
        tau : array-like
            A range of time points at which the WWZ transform will be
            calculated.
        verbose : bool, optional
            Turns printing of information on or off. The default is True.

        Returns
        -------
        None.

        Notes
        -----
        It is recommended to use the get_tau() method to create the array of
        taus.
        """

        self.tau = np.asarray(tau)
        self.len0 = tau.size

        if verbose:
            print('Tau (time points) set.')

    #--------------------------------------------------------------------------
    def transform(self, c=1./(8.*np.pi**2), snr_weights=False, verbose=0):
        """Perform the WWZ transform as defined in [1].

        Parameters
        ----------
        c : float, optional
            The window decay parameter. See [1], Sec. 1 for a description.
            The default is 1./(8.*np.pi**2).
        snr_weights : bool, optional
            If True, the fluxes are additionally weighted by their
            corresponding signal-to-noise ratio, derived from the provided
            flux uncertainties. The default is False.
        verbose : int, optional
            Defines how much information is printed out. 0, no information is
            printed. 1, some information is printed. 2, all information is
            printed. The default is 0.

        Returns
        -------
        bool
            Returns True, if the analysis was performed succesfully. False,
            otherwise.

        References
        ----------
        [1] Foster, 1996
            https://ui.adsabs.harvard.edu/abs/1996AJ....112.1709F/abstract

        """

        if self.freq is None or self.tau is None:
            print('Frequencies and/or taus are not set. Analysis aborted.')
            return False

        if verbose:
            t_start = datetime.now()
            print('Starting the WWZ transform..')

        self.c = c
        self.snr_weights = snr_weights

        if verbose > 1:
            print('Setting frequency-tau-grid..')

        # create the wavelet grid:
        freq, tau = np.meshgrid(2*np.pi*self.freq, self.tau)
        # freq: the frequencies at which to evaluate the wavelet
        # tau: the delays over which to calculate the wavelet

        if verbose > 1:
            print('Creating weights..')

        # arrays which are used but not returned
        s = np.zeros((3, 3, self.len0, self.len1), dtype=np.float64)
        s_inv = np.zeros((3, 3, self.len0, self.len1), dtype=np.float64)
        w = np.zeros((self.n, self.len0, self.len1), dtype=np.float64)

        p1=0.0
        p2=0.0
        p3=0.0
        w_d=0.0
        w_d_s=0.0
        v_x=0.0
        v_x_s=0.0
        v_y=0.0
        v_y_s=0.0

        if verbose > 1:
            print('Calculating projections and scattering matrix..')

        for a in np.arange(0, self.n):
            # weights:
            w[a] = np.exp(-self.c * freq**2 * (self.t[a] - tau)**2)

            if snr_weights:
                w[a] *= np.sqrt(self.x[a]**2 / self.s_x[a]**2)

            w_d += w[a]
            w_d_s += w[a]**2

            # weighted variation of the data:
            v_x += w[a] * self.x[a]
            v_x_s += w[a] * self.x[a]**2

            # trial functions:
            t1 = 1.0
            t2 = np.cos(freq * (self.t[a] - tau))
            t3 = np.sin(freq * (self.t[a] - tau))

            # projections:
            p1 += w[a] * t1 * self.x[a]
            p2 += w[a] * t2 * self.x[a]
            p3 += w[a] * t3 * self.x[a]

            # s-matrix elements:
            s[0,0] += w[a] * t1 * t1
            s[1,0] += w[a] * t2 * t1
            s[0,1] += w[a] * t1 * t2
            s[2,0] += w[a] * t3 * t1
            s[0,2] += w[a] * t1 * t3
            s[1,1] += w[a] * t2 * t2
            s[2,1] += w[a] * t3 * t2
            s[1,2] += w[a] * t2 * t3
            s[2,2] += w[a] * t3 * t3

        # the projections of the trial functions onto the data are:
        p1 /= w_d
        p2 /= w_d
        p3 /= w_d
        # the scattering matrix elements are:
        s /= w_d

        if verbose > 1:
            print('Inverting scattering matrix..')

        # invert scattering matrix:
        for m in np.arange(0, self.len0):
            for n in np.arange(0, self.len1):
                s_inv[:,:,m,n] = np.linalg.inv(s[:,:,m,n])

        if verbose > 1:
            print('Calculating model coefficients..')

        # model coefficients:
        y1 = s_inv[0,0] * p1 + s_inv[0,1] * p2 + s_inv[0,2] * p3
        y2 = s_inv[1,0] * p1 + s_inv[1,1] * p2 + s_inv[1,2] * p3
        y3 = s_inv[2,0] * p1 + s_inv[2,1] * p2 + s_inv[2,2] * p3

        if verbose > 1:
            print('Calculating elements of transform..')

        # calculate elements of the transform:
        for a in np.arange(self.n):
            ya = y1 * 1.0 + y2 * np.cos(freq * (self.t[a] - tau)) \
                    + y3 * np.sin(freq * (self.t[a] - tau))
            v_y += w[a] * ya
            v_y_s += w[a] * ya**2

        if verbose > 1:
            print('Calculating WWZ and WWA..')

        self.v_x = (v_x_s / w_d) - (v_x / w_d)**2
        self.v_y = (v_y_s / w_d) - (v_y / w_d)**2
        self.n_eff = w_d**2 / w_d_s
        self.wwz = ((self.n_eff - 3.) * self.v_y) \
                / (2. * (self.v_x - self.v_y))
        self.wwa = np.sqrt(y2**2 + y3**2)

        if verbose:
            t_used = datetime.now() - t_start
            print('Finished in', t_used)

        return True

    #--------------------------------------------------------------------------
    def save(self, filename):
        """Save the class instance in a python pickle file.

        Parameters
        ----------
        filename : str
            Filename under which the class instance is saved.

        Returns
        -------
        None.
        """

        with open(filename, mode='wb') as f:
            pickle.dump(self, f)

    #--------------------------------------------------------------------------
    def load(self, filename):
        """Load a saved WWZ class instance from a python pickle file.

        Parameters
        ----------
        filename : str
            File name of the pickle file.

        Returns
        -------
        WWZ-type
            Returns a instance of WWZ.
        """

        with open(filename, mode='rb') as f:
            return pickle.load(f)

    #--------------------------------------------------------------------------
    def estimate_significance(self, simulations, append=False):
        """Monte Carlo based significance estimation.

        Parameters
        ----------
        simulations : list
            Each list entry is one simulation that will be analysed in the same
            way as the real data. The data structure for each simulation need
            to be the following. A list, tuple, or numpy.ndarray, where the
            first element/row contains the time steps, the second element/row
            contains the signal. Optionally a third element contains the
            corresponding uncertainties. The uncertainty can also be a single
            value.
        append : bool, optional
            If False, currently available significance estimations will be
            discarded and overwritten with the new analysis. If True, the
            analysis of the new simulations will be added to the existing ones.
            The default is False.

        Returns
        -------
        bool
            True, if the analysis finished. False, if the analysis was aborted.

        Notes
        -----
        When new simulations are appended, i.e. append=True, it is up to the
        user to ensure that the new simulations are independent of the previous
        ones. This method does not track simulations and cannot guarantee that
        repetitions are discarded.
        """

        if self.wwz is None:
            print('Analyse data first. Aborted!')
            return False

        # reset/initialize simulation results:
        if not append or self.wwz_sig is None:
            self.wwz_sig = np.zeros(shape=self.wwz.shape)
            self.wwa_sig = np.zeros(shape=self.wwa.shape)
            self.n_sig = 0

        # prepare WWZ instance for analysis of simulations:
        analyser = WWZ()

        n = len(simulations)
        t_start = datetime.now()
        print('Starting analysis of simulations at', t_start)

        # iterate through simulations:
        for i, simulation in enumerate(simulations):
            sys.stdout.write('\rProgress: {0:d} of {1:d}. {2:.1f}%'. format(
                    i+1, n, i*100./n))

            # extract simulation data:
            time = simulation[0]
            flux = simulation[1]
            try:
                flux_err = simulation[2]
            except:
                flux_err = 0

            # analyse simulation data:
            analyser.set_data(time, flux, flux_err, verbose=False)
            analyser.set_freq(self.freq, verbose=False)
            analyser.set_tau(self.tau, verbose=False)
            analyser.transform(
                    c=self.c, snr_weights=self.snr_weights, verbose=0)
            self.wwz_sig += analyser.wwz > self.wwz
            self.wwa_sig += analyser.wwa > self.wwa
            self.n_sig += 1

        self.wwz_pval = self.wwz_sig / self.n_sig
        self.wwa_pval = self.wwa_sig / self.n_sig

        t_used = datetime.now() - t_start
        print('\rFinished in', t_used)

        return True

    #--------------------------------------------------------------------------
    def _find_peaks(self, x, y, threshold, sampling=10):
        """Find peaks above a given threshold along a 1d-signal y(x).

        Parameters
        ----------
        x : numpy.ndarray
            Positional data of the signal, e.g. frequency or period.
        y : numpy.ndarray
            Signal, e.g. WWZ, WWA, or p-value along one time bin.
        threshold : float
            Defines the noise level. Only signals above this threshold are
            detected.
        sampling : int
            The interpolated signal is evalued at n times the sampling of the
            original data, where n is set by this number.

        Returns
        -------
        peak_x : list
            Peak positions.
        peak_y : list
            Peak signal strenghts.

        Notes
        -----
        This method uses spline interpolation to improve the estimation of the
        peak position and signal. Multiple peaks are identified recursively
        starting with the strongest peak.
        """

        if np.all(y < threshold) or x.size <= 3:
            return [], []

        # the spline fit requires an increasing order of x,
        # reverse order, if x is decreasing:
        if x.size > 1 and x[0] > x[1]:
            x = x[::-1]
            y = y[::-1]

        # spline interpolation:
        tck = splrep(x, y, s=0)
        n = x.size * int(sampling)
        x_interp = np.linspace(x[0], x[-1], n)
        y_interp = splev(x_interp, tck)
        y_der1 = splev(x_interp, tck, der=1)

        # identify peak:
        i = np.argmax(y_interp)

        # maximum at data edge, no peak found:
        if i + 1 == n or i == 0:
            return [], []

        # interpolate peak position:
        f = interp1d(y_der1[i-1:i+2], x_interp[i-1:i+2])
        peak_x = float(f(0))
        peak_y = float(splev(peak_x, tck))

        # find left peak edge:
        i = np.argmax(y)
        peak_max = peak_y
        n = x.size
        for n in range(i+1, x.size):
            if y[n] < peak_max:
                peak_max = y[n]
            else:
                break

        # find right peak edge:
        peak_max = peak_y
        m = 0
        for m in range(i-1, 0, -1):
            if y[m] < peak_max:
                peak_max = y[m]
            else:
                break

        peak_x = [peak_x]
        peak_y = [peak_y]

        # recursion on left part:
        if m > 1 and np.any(y[:m] > threshold):
            peak_x_l, peak_y_l = self._find_peaks(
                    x[:m], y[:m], threshold, sampling=sampling)
            peak_x = peak_x_l + peak_x
            peak_y = peak_y_l + peak_y

        # recursion on right part:
        if n < x.size-1 and np.any(y[n:] > threshold):
            peak_x_r, peak_y_r = self._find_peaks(
                    x[n:], y[n:], threshold, sampling=sampling)
            peak_x = peak_x + peak_x_r
            peak_y = peak_y + peak_y_r

        return peak_x, peak_y

    #--------------------------------------------------------------------------
    def find_peaks(
            self, signal, quantile, sampling=10, period_space=None,
            verbose=True):
        """Find peaks along frequency/period in a given signal for all time
        bins.

        Parameters
        ----------
        signal : string
            Select 'wwz' or 'wwa'.
        quantile : float
            Defines the noise level. Only signals above the threshold are
            detected that corresponds to the set quantile.
        sampling : int
            The interpolated signal is evalued at n times the sampling of the
            original data, where n is set by this number.
        verbose : bool, optional
            Turns printing of information on or off. The default is True.

        Raises
        ------
        ValueError
            Raised when 'signal' is not set to one of the allow values.

        Returns
        -------
        peak_tau : numpy.ndarray
            Time bins with detected signal above the threshold.
        peak_freq : numpy.ndarray
            Peak position, i.e. either frequency or period, depending on
            whether
            peak_ : numpy.ndarray

        Notes
        -----
        This method iterates through all time bins. The peak identification is
        implemented in the helper method self._find_peak().

        TBD
        ----
        Analysis of the p-values is currently turned off. The current peak
        location algorith does not work with flat p-value curves, due to an
        insufficient number of simulations.
        """

        # TODO: p-value analysis deactivated because it is running into bugs.

        peak_tau = []
        peak_pos = []
        peak_signal = []

        # select signal for analysis:
        if signal not in ['wwa', 'wwz']:
            raise ValueError(
                    "For 'signal' select 'wwz' or 'wwa'.")
        y = eval(f'self.{signal}')

        # invert p-values:
        if signal.find('pval') > -1:
            y = 1. - y

        # set threshold for signal detection:
        threshold = np.quantile(y, quantile)

        if verbose:
            print('Finding peaks in {0:s}.'.format(signal.upper()))
            print(f'Threshold set to: {threshold}')

        if (period_space is None and self.linear_period) or period_space:
            x = 1. / self.freq[::-1]
            y = y.transpose()[::-1].transpose()
            if verbose:
                print('Analysis in period space.')
        else:
            x = self.freq
            if verbose:
                print('Analysis in frequency space.')

        for i, t in enumerate(self.tau):
            peak_x, peak_y = self._find_peaks(
                    x, y[i], threshold, sampling=sampling)
            for px, py in zip(peak_x, peak_y):
                peak_tau.append(t)
                peak_pos.append(px)
                peak_signal.append(py)

        peak_tau = np.array(peak_tau)
        peak_pos = np.array(peak_pos)
        peak_signal = np.array(peak_signal)

        if verbose:
            print('Done.')

        return peak_tau, peak_pos, peak_signal

#==============================================================================
