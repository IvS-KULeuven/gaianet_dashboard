from pathlib import Path
from typing import Any
import logging
import numpy as np
import polars as pl
from preprocess import preprocess_coordinates

logger = logging.getLogger(__name__)


def in_bounds_expr(bounds: tuple[float, float, float, float],
                   xdim: str,
                   ydim: str):
    x_bounds = (pl.col(xdim) > bounds[0]) & (pl.col(xdim) < bounds[2])
    y_bounds = (pl.col(ydim) > bounds[1]) & (pl.col(ydim) < bounds[3])
    return x_bounds & y_bounds


short_names = {
    'ACEP|BLHER|CEP|CW|DCEP|RV|T2CEP': 'CEP',
    'ACV|CP|MCP|ROAM|ROAP|SXARI': 'ACV',
    'ACYG': 'ACYG',
    'ARRD|RRAB|RRC|RRD': 'RR',
    'BCEP': 'BCEP',
    'BE|GCAS|SDOR|WR': 'BE',
    'BY|FLARES|ROT|SOLAR_LIKE': 'SOLAR',
    'CTTS|GTTS|HAEBE|DIP|FUOR|PULS_PMS|TTS|UXOR|WTTS|YSO': 'YSO',
    'CV': 'CV',
    'DSCT|DSCT+GDOR|GDOR|SXPHE': 'DSCT|GDOR',
    'EA|EB|ECL|EW': 'ECL',
    'EHM_ZZA|ELM_ZZA|GWVIR|HOT_ZZA|V777HER|ZZ|ZZA': 'WD',
    'ELL': 'ELL',
    'EP': 'EP',
    'LPV|LSP|M|OSARG|SARG|SR|SRA|SRB|SRC|SRD': 'LPV',
    'MICROLENSING': 'ULENS',
    'QSO': 'QSO',
    'RCB': 'RCB',
    'RS': 'RS',
    'SPB': 'SPB',
    'SYST|ZAND': 'SYST|ZAND',
    'V1093HER|V361HYA': 'V1093HER|V361HYA',
}


class Embedding:

    def __init__(self,
                 latent_dir: Path,
                 metadata: Path,
                 class_column: str = 'macro_class'):
        self.class_column = class_column
        df_emb = pl.scan_parquet(
            latent_dir / '*.parquet'
        ).rename(
            {'source_id': 'sourceid'}
        ).select(
            ['sourceid', r'^embedding_.$', 'label']
        )
        df_source = pl.scan_parquet(
            metadata
        ).select(
            ['sourceid', 'macro_class', 'class', 'L', 'B', 'in_andromeda_survey']
        )
        #.with_columns(
        #    pl.col('macro_class').replace_strict(short_names, default='none')
        #)
        df_emb = df_emb.join(df_source, on='sourceid', how='left').collect()
        self.emb_columns = [c for c in df_emb.columns if 'embedding_' in c]
        self.available_classes = df_emb.filter(
            pl.col(self.class_column).ne('UNKNOWN')
        ).select(
            self.class_column
        ).unique().sort(self.class_column).to_series().to_list()
        self.emb = df_emb
        self.find_sids_in_box(box_bounds=(-1, -1, 1, 1))

    def get_2d_embedding(self,
                         sids: np.ndarray | None = None,
                         class_name: str | None = None,
                         x_dim: str = 'embedding_0',
                         y_dim: str = 'embedding_1'
                         ) -> dict[str, np.ndarray]:
        emb = self.emb
        if sids is not None:
            emb = emb.filter(pl.col('sourceid').is_in(sids))
        else:
            sids = emb.select(['sourceid']).to_series().to_numpy()
        if class_name is not None:
            emb = emb.filter(pl.col(self.class_column).eq(class_name))
        x, y = emb.select([x_dim, y_dim]).to_numpy().T
        return {'x': x, 'y': y, 'sourceid': sids}

    def validate_selection(self,
                           requested_sids: list[int]
                           ) -> np.ndarray:
        emb = self.emb.filter(pl.col('sourceid').is_in(requested_sids))
        if len(emb) != len(requested_sids):
            logger.warning((f"Requested {len(requested_sids)} sids "
                            f"but only {len(emb)} are available"))
        sids = emb.select('sourceid').to_series()
        return sids.to_numpy()

    def find_sids_in_box(self,
                         box_bounds: tuple[float, float, float, float],
                         subsample: int = 10000,
                         x_dim: str = 'embedding_0',
                         y_dim: str = 'embedding_1',
                         ) -> np.ndarray:
        sids = self.emb.filter(
            in_bounds_expr(box_bounds, x_dim, y_dim)
        ).select('sourceid').to_series()
        if len(sids) > subsample:
            sids = sids.sample(subsample)
        return sids.to_numpy()

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
