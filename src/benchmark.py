import argparse
from pathlib import Path
import time
import pickle as pkl
import polars as pl
import numpy as np
# from data_loader import DataLoader

lc_schema = {'sourceid': pl.Int64,
             'g_obstimes': pl.List(pl.Float64),
             'g_val': pl.List(pl.Float64),
             'g_valerr': pl.List(pl.Float64)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Panel')
    parser.add_argument('data_dir', type=str)
    # parser.add_argument('metadata_path', type=str)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    # metadata_path = Path(args.metadata_path)
    cols = ['sourceid', 'g_obstimes', 'g_val', 'g_valerr']
    # dl = DataLoader(data_dir, metadata_path)
    with open(data_dir / 'light_curves' / 'index.pkl', 'rb') as f:
        lc_index = pkl.load(f)
    lc_dir = data_dir / 'light_curves'

    def load_lc1(sid):
        row, file = lc_index[sid]
        lc = pl.scan_parquet(
            lc_dir / file, schema=lc_schema,
        ).select(cols).slice(row, 1).collect()
        return lc

    def load_lc2(sid):
        _, file = lc_index[sid]
        lc = pl.scan_parquet(
            lc_dir / file, schema=lc_schema,
        ).select(cols).filter(
            pl.col('sourceid').eq(sid)
        ).collect()
        return lc

    sids = np.array(list(lc_index.keys()))
    sids = sids[np.random.permutation(len(sids))][:12]
    init = time.time()
    res1 = [load_lc1(sid) for sid in sids]
    print(f'{1000*(time.time() - init):0.4f} [ms]')

    init = time.time()
    res2 = [load_lc2(sid) for sid in sids]
    print(f'{1000*(time.time() - init):0.4f} [ms]')
