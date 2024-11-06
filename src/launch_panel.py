from pathlib import Path
import argparse
import json
import logging
import holoviews as hv
import panel as pn
import geoviews as gv

from data_loader import DataLoaderSQLite
from metadata import MetadataHandler
from dashboard import build_dashboard

logger = logging.getLogger(__name__)


FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@pn.cache
def reconfig_basic_config(format_=FORMAT, level=logging.INFO):
    """(Re-)configure logging"""
    logging.basicConfig(format=format_, level=level, force=True)
    logging.info("Logging.basicConfig completed successfully")


if __name__.startswith("bokeh"):
    reconfig_basic_config()
    parser = argparse.ArgumentParser(description='Panel')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('latent_path', type=str)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    latent_path = Path(args.latent_path)
    # More details about the classes at:
    # https://www.aanda.org/articles/aa/full_html/2023/06/aa45591-22/aa45591-22.html
    with open(data_dir / 'class_names.json', 'r') as f:
        class_description = json.load(f)
    if 'plotter' in pn.state.cache:
        plotter = pn.state.cache['plotter']
    else:
        pn.state.cache['plotter'] = plotter = DataLoaderSQLite(
            data_dir,
        )
    if 'emb' in pn.state.cache:
        emb = pn.state.cache['emb']
    else:
        pn.state.cache['emb'] = emb = MetadataHandler(
            latent_path,
            data_dir / 'features.parquet',
            data_dir / 'metadata'
        )
    pn.extension(loading_indicator=True,
                 loading_spinner='dots',
                 notifications=True,
                 global_loading_spinner=True)
    hv.extension('bokeh')
    gv.extension('bokeh')
    dashboard = build_dashboard(plotter, emb, class_description,
                                n_cols=3, n_rows=3)
    dashboard.servable(title='GaiaNet Embedding Explorer')
