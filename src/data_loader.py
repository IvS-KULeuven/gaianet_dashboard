from pathlib import Path
import logging
import pickle
import numpy as np
import polars as pl
import pandas as pd
import holoviews as hv
import h5py
from gaiaxpy import convert

from silencer import suppress_print

logger = logging.getLogger(__name__)


def create_index(path_to_data: Path) -> dict[str, str]:
    index = {}
    for p in path_to_data.glob('*.h5'):
        with h5py.File(p, 'r') as f:
            sids = list(f.keys())
        for sid in sids:
            index[sid] = p.name
    return index


def load_index(path_to_index: Path) -> dict[str, str]:
    if path_to_index.exists():
        logger.info('Found h5 index')
        with open(path_to_index, 'rb') as f:
            index = pickle.load(f)
    else:
        logger.info('h5 index not found, building it from scratch')
        index = create_index(path_to_index.parent)
        with open(path_to_index, 'wb') as f:
            pickle.dump(index, f)
    return index


class DataLoader():

    def __init__(self,
                 dataset_dir: Path,
                 metadata_path: Path,
                 bands: list[str] = ['g']):
        self.lc_dir = dataset_dir / 'light_curves_hdf5'
        self.lc_index = load_index(self.lc_dir / 'index.pkl')
        logger.info(f'Found {len(self.lc_index)} sources with light curves')
        self.xp_dir = dataset_dir / 'reduced_spectra_hdf5'
        self.xp_index = load_index(self.xp_dir / 'index.pkl')
        logger.info(f'Found {len(self.lc_index)} sources with xp spectra')
        self.features = pl.read_parquet(dataset_dir / 'features' / '*.parquet')
        metadata = pl.read_parquet(metadata_path).rename({'sourceid': 'source_id'})
        self.features = self.features.join(metadata, on='source_id', how='left')
        feature_names = self.features.columns
        self.available_periods = [c for c in feature_names if 'frequency' in c]
        self.selected_frequency = 'MHAOV_best_frequency'
        self.lc_cols = ['sourceid']
        for band in bands:
            for col in ['obstimes', 'val', 'valerr']:
                self.lc_cols.append(f'{band}_{col}')

    def get_features(self, sids: list[int]) -> pl.DataFrame:
        if len(sids) == 1:
            filter_expr = pl.col('source_id').eq(sids[0])
        else:
            filter_expr = pl.col('source_id').is_in(sids)
        return self.features.filter(filter_expr)

    def get_frequency(self, sid: int) -> float:
        return self.features.lazy().filter(
            pl.col('source_id').eq(sid)
        ).select(self.selected_frequency).collect().item()

    def plot_features(self,
                      sids: list[int],
                      width: int = 350,
                      height: int = 250):
        selection = self.get_features(sids)
        plots = []

        def kde_plot(data, label, bw=0.05):
            return hv.Distribution(
                data, kdims=label
            ).opts(width=width, height=height, framewise=True, bandwidth=bw)

        feature = selection.select('magnitude_mean').to_numpy()
        label = 'G-band mean'
        plots.append(kde_plot(feature, label))
        feature = selection.select('magnitude_std').to_numpy()
        label = 'G-band standard deviation'
        plots.append(kde_plot(feature, label))
        feature = selection.select(self.selected_frequency).to_numpy()
        feature = np.log10(feature)
        # TODO: ADD COLOR
        label = 'Log10 dominant frequency'
        plots.append(kde_plot(feature, label))
        # Top 5 classes
        c = selection.select('class').filter(
            pl.col('class').ne('UNKNOWN')
        ).group_by('class').count().top_k(5, by='count')
        plots.append(
            hv.Bars(
                c.to_pandas(), kdims=['class'], vdims=['count']
            ).opts(width=350, height=250)
        )

        return hv.Layout(plots).cols(2).opts(shared_axes=False)

    def get_lightcurve(self,
                       sid: str | int,
                       pack_kwargs: dict = {}) -> dict[str, np.ndarray]:
        if isinstance(sid, int):
            sid = str(sid)
        if sid not in self.lc_index:
            raise ValueError(f"No light curves found for source {sid}")
        file = self.lc_index[sid]
        with h5py.File(self.lc_dir / file, 'r') as f:
            lc = f[sid][:]
        return {'g': lc}

    def get_continuous_spectra(self,
                               sid: str | int,
                               ) -> dict[str, np.ndarray]:
        if isinstance(sid, int):
            sid = str(sid)
        if sid not in self.xp_index:
            raise ValueError(f"No Xp spectra found for source {sid}")
        file = self.xp_index[sid]
        with h5py.File(self.xp_dir / file, 'r') as f:
            xp = f[sid][:]
        return {'bp': xp[0], 'rp': xp[1]}

    def get_sampled_spectra(self,
                            sid: str | int,
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
                        sid: str,
                        width: int = 250,
                        height: int = 160,
                        plot_errors: bool = True,
                        folded: bool = False,
                        **kwargs):
        if sid is not None:
            lc = self.get_lightcurve(sid)
            time, mag, err = lc['g']
            title = sid
            if folded:
                # best_freq = estimate_dominant_frequency(lc, multiband=False)
                best_freq = self.get_frequency(int(sid))
                P = 2.0/best_freq
                time = np.mod(time, P)/P
                title = title+f' f={best_freq:5.5g}'
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
        return hv.Overlay(
            plots
        ).opts(shared_axes=True, title=title, fontsize={'title': 8})

    def plot_spectra(self,
                     sid: str,
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
