import time
import logging
from pathlib import Path
import numpy as np
import polars as pl
import holoviews as hv


logger = logging.getLogger(__name__)

short_names = {
    'ACEP|BLHER|CEP|CW|DCEP|RV|T2CEP': 'CEP',
    'ACV|CP|MCP|ROAM|ROAP|SXARI': 'ACV',
    # 'ACYG': 'ACYG',
    'ARRD|RRAB|RRC|RRD': 'RR',
    # 'BCEP': 'BCEP',
    'BE|GCAS|SDOR|WR': 'BE',
    'BY|FLARES|ROT|SOLAR_LIKE': 'SOLAR',
    'CTTS|GTTS|HAEBE|DIP|FUOR|PULS_PMS|TTS|UXOR|WTTS|YSO': 'YSO',
    'CV': 'CV',
    'DSCT|DSCT+GDOR|GDOR|SXPHE': 'DSCT|GDOR',
    'EA|EB|ECL|EW': 'ECL',
    'EHM_ZZA|ELM_ZZA|GWVIR|HOT_ZZA|V777HER|ZZ|ZZA': 'WD',
    'ELL': 'ELL',
    # 'EP': 'EP',
    'LPV|LSP|M|OSARG|SARG|SR|SRA|SRB|SRC|SRD': 'LPV',
    # 'MICROLENSING': 'ULENS',
    'QSO': 'QSO',
    # 'RCB': 'RCB',
    'RS': 'RS',
    # 'SPB': 'SPB',
    # 'SYST|ZAND': 'SYST|ZAND',
    #'V1093HER|V361HYA': 'V1093HER|V361HYA',
}


def filter_sids_expression(sids: list[int]) -> pl.Expr:
    if len(sids) == 1:
        filter_expr = pl.col('source_id').eq(sids[0])
    else:
        filter_expr = pl.col('source_id').is_in(sids)
    return filter_expr


class MetadataHandler:

    def __init__(self,
                 embedding_path: Path,
                 features_path: Path,
                 meta_dir: Path,
                 ):
        embedding = pl.scan_csv(
            embedding_path
        ).select(
            ['source_id', r'^embedding_.$']
        )
        meta = pl.scan_csv(
            meta_dir / 'meta.csv'
        ).select(
            ['source_id', 'ruwe', 'bp_rp', 'longitude', 'latitude']
        )
        labels = pl.scan_csv(
            meta_dir / 'labels.csv'
        ).select(
            ['source_id', 'class', 'macro_class']
        )
        sos_labels = pl.scan_csv(
            meta_dir / 'labels_sos.csv'
        ).select(
            ['source_id', 'SOS_class']
        )  # .rename({'SOS_subclass': 'class', 'SOS_class': 'macro_class'})
        features = pl.scan_parquet(features_path)
        self.metadata = features.join(
            meta, on='source_id',
        ).join(
            embedding, on='source_id'
        ).join(
            labels, on='source_id', how='left'
        ).join(
            sos_labels, on='source_id', how='left'
        ).collect()
        self.metadata_hv = hv.Dataset(self.metadata.to_pandas())
        self.labeled_metadata = self.metadata.filter(
            ~pl.col('macro_class').is_null() & pl.col('macro_class').ne('UNKNOWN')
        ).with_columns(
            pl.col('macro_class').replace_strict(short_names, default=None)
        ).drop_nulls().sort(by='macro_class')
        print(self.metadata.shape)
        meta_cols = self.metadata.columns
        self.available_periods = [c for c in meta_cols if 'frequency' in c]
        self.selected_frequency = 'NUFFT_best_frequency'
        self.emb_columns = [c for c in meta_cols if 'embedding_' in c]
        sos_names = self.metadata.select(
            'SOS_class'
        ).drop_nulls().group_by('SOS_class').len().with_columns(
            pl.concat_str([
                pl.col("SOS_class"),
                pl.concat_str([
                    pl.lit("("), pl.col("len").cast(pl.Utf8), pl.lit(")")
                ], separator="").alias("len")
            ], separator=" ").alias("class_with_len"),
        ).sort('SOS_class').select(['class_with_len', 'SOS_class']).to_dict(as_series=False)
        self.sos_classes = {v1: v2 for v1, v2 in zip(sos_names['class_with_len'], sos_names['SOS_class'])}

    def get_all_metadata(self, sids: list[int]) -> pl.DataFrame:
        return self.metadata.filter(filter_sids_expression(sids))

    def get_frequencies(self, sids: list[int]) -> np.ndarray:
        frequencies = self.metadata.filter(
            filter_sids_expression(sids)
        ).select(self.selected_frequency).to_series().to_numpy()
        return frequencies

    def filter_embedding(self,
                         sids: np.ndarray | None = None,
                         class_name: str | None = None,
                         class_column: str = 'class',
                         ) -> pl.DataFrame:
        tinit = time.time()
        emb = self.metadata
        if sids is not None:
            emb = emb.filter(pl.col('source_id').is_in(sids))
        if class_name is not None:
            emb = emb.filter(pl.col(class_column).eq(class_name))
        logger.info(f"Filter 2d embedding: {time.time()-tinit:0.4f}")
        return emb

    def filter_embedding_bound(self,
                               bounds: tuple[float, float, float, float]
                               ) -> pl.DataFrame:
        expr = (pl.col('embedding_0') > bounds[0]) & (pl.col('embedding_0') < bounds[2]) & (pl.col('embedding_1') > bounds[1]) & (pl.col('embedding_1') < bounds[3])
        return self.metadata.filter(expr)

    def validate_uploaded_sources(self, sids_text) -> pl.DataFrame | None:
        try:
            df_upload = pl.read_csv(
                sids_text, schema={'source_id': pl.Int64},
                has_header=False, new_columns=['source_id']
            )
        except Exception as e:
            logger.error(f"Failed to parse uploaded file: {e}")
            return None
        return df_upload.join(self.metadata, on='source_id')
