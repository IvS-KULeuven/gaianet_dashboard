from pathlib import Path
import logging
import pickle
import numpy as np
import h5py

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

    """
    This class loads and returns light curves, spectra and dmdts
    upon request from the interface
    """

    def __init__(self,
                 dataset_dir: Path,
                 load_lightcurves: bool = True,
                 load_xpspectra: bool = True,
                 load_dmdt: bool = True,
                 ):
        if load_lightcurves:
            self.lc_dir = dataset_dir / 'light_curves_hdf5'
            self.lc_index = load_index(self.lc_dir / 'index.pkl')
            logger.info(f'Found {len(self.lc_index)} light curves')
        else:
            self.lc_index = None
        if load_xpspectra:
            self.xp_dir = dataset_dir / 'reduced_spectra_hdf5'
            self.xp_index = load_index(self.xp_dir / 'index.pkl')
            logger.info(f'Found {len(self.xp_index)} xp spectra')
        else:
            self.xp_index = None
        if load_dmdt:
            self.dmdt_dir = dataset_dir / 'dmdt_hdf5'
            self.dmdt_index = load_index(self.dmdt_dir / 'index.pkl')
            logger.info(f'Found {len(self.dmdt_index)} dmdts')
        else:
            self.dmdt_index = None

    def get_dataproduct(self,
                        sid: str | int,
                        directory: Path,
                        index: dict | None
                        ) -> np.ndarray:
        if index is None:
            raise ValueError("The requested data product was not loaded")
        if isinstance(sid, int):
            sid = str(sid)
        if sid not in index:
            raise ValueError(f"No data product found for source {sid}")
        file = index[sid]
        with h5py.File(directory / file, 'r') as f:
            data = np.array(f[sid][:])
        return data

    def get_lightcurve(self,
                       sid: str | int,
                       ) -> np.ndarray:
        return self.get_dataproduct(sid, self.lc_dir, self.lc_index)

    def get_spectra(self,
                    sid: str | int,
                    ) -> np.ndarray:
        return self.get_dataproduct(sid, self.xp_dir, self.xp_index)

    def get_dmdt(self,
                 sid: str | int,
                 ) -> np.ndarray:
        return self.get_dataproduct(sid, self.dmdt_dir, self.dmdt_index)
