import numpy as np
import polars as pl
from scipy.stats import gaussian_kde
# from KDEpy import FFTKDE


def kde(data,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        bandwidth: str | float = 0.1,
        resolution: int = 256,
        ) -> tuple[np.ndarray, np.ndarray]:
    kernel = gaussian_kde(data, bw_method=bandwidth)
    if lower_bound is None:
        lower_bound = np.amin(data)
    if upper_bound is None:
        upper_bound = np.amax(data)
    x = np.linspace(lower_bound, upper_bound, resolution)
    y = kernel(x)
    # x, y = FFTKDE(bw=bandwidth).fit(data).evaluate(resolution)
    return x, y


class UserSelection:
    def __init__(self):
        self.has_data = False
        self.sids = []
        self.embedding = []
        self.freqs = []
        self.coordinates = []
        self.labels_sos = []
        self.labels_cu7 = []
        self.features = {}
        self.label_count = {}

    def fill_selection(self, selection: pl.DataFrame):
        if len(selection) == 0:
            self.has_data = False
            return
        self.has_data = True
        self.sids = selection['source_id'].to_numpy()
        self.embedding = selection[['embedding_0', 'embedding_1']].to_numpy()
        self.freqs = selection['NUFFT_best_frequency'].to_numpy()
        self.labels_sos = selection['SOS_class'].to_numpy()
        self.labels_cu7 = selection['class'].to_numpy()
        self.coordinates = selection[['longitude', 'latitude']].to_numpy()
        for feature_name in ['magnitude_mean', 'magnitude_std', 'bp_rp', 'NUFFT_best_frequency']:
            data = selection[feature_name].to_numpy()
            if 'frequency' in feature_name:
                data = -np.log10(data)
            self.features[feature_name] = kde(data)

    def random_sample(self,
                      n: int
                      ) -> tuple[list, list, list] | None:
        if not self.has_data:
            return None
        sids = self.sids
        freqs = self.freqs
        labels = self.labels_cu7
        if len(sids) > n:
            perm = np.random.permutation(len(sids))[:n]
            sids = sids[perm]
            freqs = freqs[perm]
            labels = labels[perm]
        sids = sids.tolist()
        freqs = freqs.tolist()
        labels = labels.tolist()
        return sids, freqs, labels