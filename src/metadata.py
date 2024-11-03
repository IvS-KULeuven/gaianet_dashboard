import time
import logging
from pathlib import Path
import numpy as np
import polars as pl
import holoviews as hv


logger = logging.getLogger(__name__)


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
                 meta_path: Path,
                 labels_path: Path,
                 ):
        embedding = pl.scan_parquet(
            embedding_path
        ).select(
            ['source_id', r'^embedding_.$']
        )
        meta = pl.scan_csv(
            meta_path
        ).select(
            ['source_id', 'ruwe', 'bp_rp', 'longitude', 'latitude']
        )
        labels = pl.scan_csv(
            labels_path
        ).select(
            ['source_id', 'class', 'macro_class']
        )
        features = pl.scan_parquet(features_path)
        self.metadata = features.join(
            meta, on='source_id',
        ).join(
            embedding, on='source_id'
        ).join(
            labels, on='source_id', how='left'
        ).collect()
        self.metadata_hv = hv.Dataset(self.metadata.to_pandas())
        print(self.metadata.shape)
        meta_cols = self.metadata.columns
        self.available_periods = [c for c in meta_cols if 'frequency' in c]
        self.selected_frequency = 'NUFFT_best_frequency'
        self.emb_columns = [c for c in meta_cols if 'embedding_' in c]

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
                         ) -> hv.Dataset:
        tinit = time.time()
        emb = self.metadata
        if sids is not None:
            emb = emb.filter(pl.col('source_id').is_in(sids))
        #else:
        #    sids = emb.select(['source_id']).to_series().to_numpy()
        if class_name is not None:
            emb = emb.filter(pl.col('class').eq(class_name))
        #x, y = emb.select([x_dim, y_dim]).to_numpy().T
        logger.info(f"Filter 2d embedding: {time.time()-tinit:0.4f}")
        return hv.Dataset(emb.to_pandas())

    def get_galactic_coordinates(self,
                                 sids: list[int] | None = None,
                                 origin: float = 0.0,
                                 ) -> tuple[np.ndarray, np.ndarray]:
        tinit = time.time()
        if sids is None:
            long, lat = self.metadata.select(['L', 'B']).to_numpy().T
        else:
            long, lat = self.metadata.filter(
                pl.col('source_id').is_in(sids)
            ).select(['L', 'B']).to_numpy().T
        long, lat = preprocess_coordinates(long, lat, origin=origin)
        logger.info(f"Filter galactic coordinates: {time.time()-tinit:0.4f}")
        return long, lat

    def plot_features(self,
                      sids: list[int],
                      width: int = 350,
                      height: int = 250):
        tinit = time.time()
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
        logger.info(f"Building feature plot: {time.time()-tinit:0.4f}")
        return hv.Layout(plots).cols(2).opts(shared_axes=False)


        
