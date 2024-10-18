import numpy as np
from astropy.timeseries import LombScargle, LombScargleMultiband


def estimate_dominant_frequency(lc,
                                multiband: bool = False,
                                fres: float = 1e-4):
    freq = np.arange(1e-3, 25.0, fres)
    if not multiband:
        time, mag, err = lc['g']
        ls = LombScargle(time, mag, err)
        power_args = {'method': 'fast'}
    else:
        time = np.concatenate([lc['g'][0], lc['bp-rp'][0]])
        mag = np.concatenate([lc['g'][1], lc['bp-rp'][1]])
        err = np.concatenate([lc['g'][2], lc['bp-rp'][2]])
        bands = np.array(['g']*len(lc['g'][0]) + ['bp-rp']*len(lc['bp-rp'][0]))
        ls = LombScargleMultiband(time, mag, bands, err)
        power_args = {'method': 'fast', 'sb_method': 'fast'}
    ampl = ls.power(freq, **power_args)
    best_freq = freq[np.argmax(ampl)]
    freq = np.arange(best_freq - fres, best_freq + fres, fres*0.1)
    ampl = ls.power(freq, **power_args)
    return freq[np.argmax(ampl)]
