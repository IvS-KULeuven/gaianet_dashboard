from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import holoviews as hv
from gaiaxpy import convert
from astropy.timeseries import LombScargle

from preprocess import pack_light_curve, pack_spectra
from silencer import suppress_print


class DataLoader():

    def __init__(self, dataset_dir: Path, bands: list[str] = ['g', 'bp-rp']):
        lc_index = {}
        self.lc_dir = dataset_dir / 'light_curves'
        for p in self.lc_dir.glob('*parquet'):
            for sid in pl.scan_parquet(p).select('sourceid').collect().to_series().to_list():
                lc_index[sid] = p.name
        xp_index = {}
        self.xp_dir = dataset_dir / 'reduced_spectra'
        for p in self.xp_dir.glob('*parquet'):
            for sid in pl.scan_parquet(p).select('sourceid').collect().to_series().to_list():
                xp_index[sid] = p.name
        self.lc_index = lc_index
        self.xp_index = xp_index
        self.lc_cols = ['sourceid']
        for band in bands:
            for col in ['obstimes', 'val', 'valerr']:
                self.lc_cols.append(f'{band}_{col}')

    def get_lightcurve(self, sid: int) -> dict[str, np.ndarray]:
        if sid not in self.lc_index:
            raise ValueError(f"No light curves found for source {sid}")
        lc_row = pl.scan_parquet(
            self.lc_dir / self.lc_index[sid]
        ).select(self.lc_cols).filter(pl.col('sourceid').eq(sid)).collect()
        return pack_light_curve(lc_row, remove_extreme_errors=True)

    def get_continuous_spectra(self, sid: int) -> dict[str, np.ndarray]:
        if sid not in self.xp_index:
            raise ValueError(f"No Xp spectra found for source {sid}")
        xp_row = pl.scan_parquet(
            self.xp_dir / self.xp_index[sid]
        ).filter(pl.col('sourceid').eq(sid)).collect()
        return pack_spectra(xp_row)

    def get_sampled_spectra(self,
                            sid: int,
                            pseudo_wavelenghts: np.ndarray | None = None,
                            ) -> dict[str, np.ndarray]:
        """
        Retrieves coefficients from disk and creates a dummy dataframe
        for convert(). Because we don't have the correlation matrix,
        the errors generated by convert are not valid, but fluxes are ok.
        """
        if pseudo_wavelenghts is None:
            pseudo_wavelenghts = np.linspace(0, 60, 100)
        xp = self.get_continuous_spectra(sid)
        bp_coef, bp_coef_err = xp['bp']
        rp_coef, rp_coef_err = xp['rp']
        n = len(bp_coef)
        fake_corr_matrix = np.eye(n)
        df_xp = pd.DataFrame({
            'source_id': sid,
            'bp_n_parameters': n,
            'rp_n_parameters': n,
            'bp_standard_deviation': np.std(bp_coef),
            'rp_standard_deviation': np.std(rp_coef),
            'bp_coefficients': [bp_coef],
            'bp_coefficient_errors': [bp_coef_err],
            'rp_coefficients': [rp_coef],
            'rp_coefficient_errors': [rp_coef_err],
            'bp_coefficient_correlations': [fake_corr_matrix],
            'rp_coefficient_correlations': [fake_corr_matrix]
        })
        df_fake = suppress_print(convert,
                                 input_object=df_xp,
                                 sampling=pseudo_wavelenghts,
                                 with_correlation=False,
                                 save_file=False)[0]
        bp_spectra = df_fake.loc[df_fake['xp'] == 'BP']['flux'].item()
        rp_spectra = df_fake.loc[df_fake['xp'] == 'RP']['flux'].item()
        return {'bp': bp_spectra, 'rp': rp_spectra}

    def plot_lightcurve(self,
                        sid: int,
                        width: int = 250,
                        height: int = 160,
                        plot_errors: bool = False,
                        folded: bool = True,
                        **kwargs):
        if sid is not None:
            time, mag, err = self.get_lightcurve(sid)['g']
            if folded:
                freq = np.arange(1e-3, 15.0, 1e-4)
                ls = LombScargle(time, mag, err)
                ampl = ls.power(
                    freq, assume_regular_frequency=True, method='fast'
                )
                best_freq = freq[np.argmax(ampl)]
                freq = np.arange(best_freq - 1e-4, best_freq + 1e-4, 1e-5)
                ampl = ls.power(
                    freq, assume_regular_frequency=True, method='fast'
                )
                P = 1.0/freq[np.argmax(ampl)]
                time = np.mod(time, P)/P
            title = str(sid)
        else:
            time, mag, err = [], [], []
            title = ''
        dots = hv.Scatter(
            (time, mag),
            kdims=['Time' if not folded else 'Phase'],
            vdims=['Magnitude']
        ).opts(color='g', framewise=True, invert_yaxis=True,
               width=width, height=height)
        plots = [dots]
        if plot_errors:
            bars = hv.ErrorBars(
                (time, mag, err),
                kdims=['Time' if not folded else 'Phase'],
                vdims=['Magnitude', 'Error']
            ).opts(color='g', lower_head=None, upper_head=None)
            plots.append(bars)
        return hv.Overlay(plots).opts(shared_axes=True,
                                      title=title,
                                      fontsize={'title': 8})

    def plot_spectra(self, sid: int,
                     width: int = 250,
                     height: int = 160,
                     pseudo_wavelenght_resolution: int = 60,
                     **kwargs):
        if sid is not None:
            pw = np.linspace(0, 60, pseudo_wavelenght_resolution)
            xp = self.get_sampled_spectra(sid, pseudo_wavelenghts=pw)
            bp, rp = xp['bp'], xp['rp']
            title = str(sid)
        else:
            pw, bp, rp = [], [], []
            title = ''
        c_bp = hv.Curve(
            (pw, bp), kdims=['Pseudo frequency'], vdims=['Flux']
        ).opts(color='b', framewise=True, width=width, height=height)
        c_rp = hv.Curve(
            (pw, rp), kdims=['Pseudo frequency'], vdims=['Flux']
        ).opts(color='r', framewise=True)
        return hv.Overlay(
            [c_bp, c_rp]
        ).opts(shared_axes=True, title=title, fontsize={'title': 8})
