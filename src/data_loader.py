from pathlib import Path
import logging
import time
import pickle
import sqlite3
import numpy as np
import h5py

logger = logging.getLogger(__name__)


class DataLoaderSQLite:

    def __init__(self, path_to_data: Path):
        self._conn = sqlite3.connect(path_to_data / 'consolidated.db')
        self.cursor = self._conn.cursor()

    def retrieve_data(self, sid: str):
        self.cursor.execute(
            "SELECT lc, xp, dmdt FROM data WHERE sid = ?", (sid,)
        )
        result = self.cursor.fetchone()
        lc = np.frombuffer(result[0])
        n = lc.shape[0]//3
        lc = lc.reshape(3, n)
        xp = np.frombuffer(result[1]).reshape(2, 2, 55)
        dmdt = np.frombuffer(result[2], dtype=np.float32).reshape(23, 5)
        return lc, xp, dmdt

    def __del__(self):
        self._conn.close()

    def profile_test(self):
        tinit = time.time()
        sids = [
            40023739377166080,
            52968800869454080,
            46618953718068992,
            46309131957199360,
            47034706552255488,
            48103564995766528,
            40330159524038912,
            37123055904340992,
            52092941077056000,
            39412960668016896,
            48200910431063680,
            52956023344987136
        ]
        for sid in sids:
            self.retrieve_data(sid)
        logger.info(f"Retrieving data: {time.time()-tinit:0.4f}")
        print(f"Retrieving data: {time.time()-tinit:0.4f}")



def load_npz(path_to_data: Path,
             sid: str | int
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path_to_data / 'products' / f'{sid}.npz')
    return data['lc'], data['xp'], data['dmdt']


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
        # file = index[sid]
        #with h5py.File(directory / file, 'r') as f:
        #    data = np.array(f[sid][:])
        data = np.load(directory / f'{sid}.npy')
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
