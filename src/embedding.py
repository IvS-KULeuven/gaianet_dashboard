from pathlib import Path
from typing import Any
import numpy as np
import polars as pl
from preprocess import preprocess_coordinates


def in_bounds_expr(bounds: tuple[float, float, float, float],
                   xdim: str,
                   ydim: str):
    x_bounds = (pl.col(xdim) > bounds[0]) & (pl.col(xdim) < bounds[2])
    y_bounds = (pl.col(ydim) > bounds[1]) & (pl.col(ydim) < bounds[3])
    return x_bounds & y_bounds


short_names = {-1: 'UNKNOWN',
               0: 'CEP', 1: 'ACV', 2: 'RR', 3: 'BCEP',
               4: 'BE', 5: 'SOLAR_LIKE', 6: 'YSO',
               7: 'CV', 8: 'DSCT|GDOR', 9: 'ECL', 10: 'WD',
               11: 'ELL', 12: 'LPV', 13: 'QSO', 14: 'RS',
               15: 'SPB', 16: 'SYST|ZAND'}


class Embedding:

    def __init__(self, latent_dir: Path):
        df_emb = pl.scan_parquet(
            latent_dir / '2' / '*.parquet'
        ).rename(
            {'source_id': 'sourceid'}
        ).select(
            ['sourceid', 'embedding_0', 'embedding_1', 'label']
        ).with_columns(
            pl.col('label').replace_strict(short_names).alias('class')
        )

        df_source = pl.scan_parquet(
            latent_dir / 'meta.parquet'
        ).select(
            ['sourceid', 'class', 'L', 'B', 'in_andromeda_survey']
        ).rename({'class': 'subclass'})
        df_emb = df_emb.join(df_source, on='sourceid', how='left').collect()
        self.emb = df_emb

    def get_embedding(self) -> dict[str, np.ndarray]:
        sids = self.emb.select(['sourceid']).to_series().to_numpy()
        x, y = self.emb.select(['embedding_0', 'embedding_1']).to_numpy().T
        return {'x': x, 'y': y, 'sourceid': sids}

    def get_labeled_embedding(self) -> dict[str, np.ndarray]:
        labeled_emb = self.emb.filter(
            pl.col('label').ne(-1)
        ).select(
            ['sourceid', 'embedding_0', 'embedding_1', 'class']
        ).sort('class')
        sids = labeled_emb.select(['sourceid']).to_series().to_numpy()
        labels = labeled_emb.select(['class']).to_series().to_numpy()
        x, y = labeled_emb.select(['embedding_0', 'embedding_1']).to_numpy().T
        return {'x': x, 'y': y, 'sourceid': sids, 'label': labels}

    def find_sids_in_box(self,
                         box_bounds: tuple[float, float, float, float],
                         subsample: int = 1000,
                         ) -> list[int]:
        sids = self.emb.filter(
            in_bounds_expr(box_bounds, 'embedding_0', 'embedding_1')
        ).select('sourceid').to_series()
        if len(sids) > subsample:
            sids = sids.sample(subsample)
        return sids.to_list()

    def get_galactic_coordinates(self,
                                 sids: list[int] | None = None,
                                 origin: float = 0.0,
                                 ) -> tuple[np.ndarray, np.ndarray]:
        if sids is None:
            long, lat = self.emb.select(['L', 'B']).to_numpy().T
        else:
            long, lat = self.emb.filter(
                pl.col('sourceid').is_in(sids)
            ).select(['L', 'B']).to_numpy().T
        return preprocess_coordinates(long, lat, origin=origin)

    def get_all_metadata(self, sid: int) -> dict[str, Any]:
        row = self.emb.filter(
            pl.col('sourceid').eq(sid)
        ).drop('label')
        if len(row) == 0:
            raise ValueError("The requested source id does not exist.")
        return row.to_dicts()[0]
