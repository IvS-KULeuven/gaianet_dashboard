import io
from pathlib import Path
import argparse
import json
from functools import partial
from typing import Callable
import logging
import numpy as np
import holoviews as hv
from holoviews import streams
import holoviews.operation.datashader as hd
import panel as pn
import geoviews as gv
from cartopy import crs

from data_loader import DataLoader
from embedding import Embedding

logger = logging.getLogger(__name__)

colors = [
    'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink',
    'olive', 'cyan', 'magenta', 'yellow', 'teal', 'gold', 'indigo', 'lime',
    'navy', 'coral', 'salmon', 'darkgreen', 'orchid', 'sienna',
    'turquoise', 'maroon', 'darkblue'
]


# TODO: MAKE DYNAMIC MAP SCALE THE AXES
def datashade_embedding(emb_plot, cnorm: str = "eq_hist"):
    return hd.dynspread(
        hd.datashade(emb_plot, cmap="gray", cnorm=cnorm),
        max_px=3, threshold=0.75, shape='circle',
    ).opts(width=650, height=500, tools=['box_select'],
           active_tools=['box_select', 'wheel_zoom'])


def build_panel(plotter: DataLoader,
                embedding: Embedding,
                class_metadata: dict,
                n_rows: int = 3,
                n_cols: int = 3):
    n_plots = n_rows*n_cols
    button_width = 100

    # Pipes for passing source ids
    sids_pipe = streams.Pipe(data=[])
    sids_pipe_plots = streams.Pipe(data=[])

    # Embedding plot
    def plot_embedding(x_dim: str,
                       y_dim: str,
                       class_name: str | None = None,
                       sids: np.ndarray | None = None,
                       **plot_kwargs):
        sel_emb = embedding.get_2d_embedding(
            sids=sids, class_name=class_name, x_dim=x_dim, y_dim=y_dim)
        return hv.Points(
            sel_emb,
            kdims=['x', 'y'],
        ).opts(**plot_kwargs)

    emb_columns = emb.emb_columns
    emb_select = partial(pn.widgets.Select, width=150, options=emb_columns)
    x_sel = emb_select(value=emb_columns[0], description='asd')
    y_sel = emb_select(value=emb_columns[1])
    emb_stream = {'x_dim': x_sel.param.value, 'y_dim': y_sel.param.value}
    bg_emb = hv.DynamicMap(plot_embedding, streams=emb_stream)

    class_sel = pn.widgets.Select(groups=class_metadata, value='none')

    def plot_class(x_dim, y_dim, name='none'):
        if name == 'none':
            return hv.Points([])
        return plot_embedding(
            class_name=name, x_dim=x_dim, y_dim=y_dim, alpha=0.5
        )
    fg_emb = hv.DynamicMap(
        plot_class,
        streams=emb_stream | {'name': class_sel.param.value}
    )

    # Source selection via textbox
    sids_input = pn.widgets.TextAreaInput(
        value="", width=300, height=180, disabled=False, max_length=500000,
        placeholder=(
            "Enter one Gaia DR3 source id per line, "
            "e.g\n2190530735119392128\n5874749936451323264\n"
            "...\nor upload them in a plain text file. "
            "Then press submit to highlight them in the embedding. "
        ), styles={'font-size': '12px'}
    )
    sids_submit_btn = pn.widgets.Button(
        name="✅ Submit", width=button_width)

    def update_plotted_sids(value=None,
                            sids: np.ndarray | None = None):
        if sids is None:
            sids = np.array(sids_pipe.data)
        if len(sids) > n_plots:
            perm = np.random.permutation(len(sids))
            sids = sids[perm][:n_plots]
        sids_plots = [str(s) for s in sids]
        if len(sids_plots) < n_plots:
            sids_plots += [None]*(n_plots-len(sids))
        sids_pipe_plots.send(sids_plots)

    def update_selection_via_textbox(value):
        sids_textbox = sids_input.value
        if not isinstance(sids_textbox, str):
            return
        if len(sids_textbox) == 0:
            return
        if sids_textbox[-1] == '\n':
            sids_textbox = sids_textbox[:-1]
        if '\n' not in set(sids_textbox):
            sids = [int(sids_textbox)]
        else:
            sids = [int(s) for s in sids_textbox.split('\n')]
        sids = embedding.validate_selection(sids)
        sids_input.value = "\n".join([str(s) for s in sids])
        sids_pipe.send(sids)
        update_plotted_sids(sids=sids)
    pn.bind(update_selection_via_textbox, value=sids_submit_btn, watch=True)

    # Copy selection to clipboard
    sids_copy_btn = pn.widgets.Button(
        name="📋 Copy", width=button_width,
        description="Copies the content of the text box to your clipboard"
    )
    sids_copy_btn.js_on_click(
        args={"source": sids_input},
        code="navigator.clipboard.writeText(source.value);"
    )

    # Download selection as CSV
    def download_sourceids():
        sids = sids_input.value
        if sids is not None:
            return io.StringIO(str(sids))
    sids_download_btn = pn.widgets.FileDownload(
        callback=download_sourceids, filename='sids.txt',
        label="⬇️  Download", width=button_width,
        description=(
            "Downloads the content of the text box as a sids.txt file"
        )
    )

    def disable_download(event):
        sids = sids_input.value
        if sids == '':
            sids_download_btn.disabled = True
            sids_clear_btn.disabled = True
        else:
            sids_download_btn.disabled = False
            sids_clear_btn.disabled = False
    pn.bind(disable_download, sids_input.param.value, watch=True)

    # Upload CSV with selection
    sids_upload_btn = pn.widgets.FileInput(
        name="📤 Upload", width=2*button_width, sizing_mode="fixed",
        multiple=False, directory=False, accept='.csv,.txt',
    )

    def parse_text_file(value):
        if not value or not isinstance(value, bytes):
            return None
        sids_input.value = value.decode("utf8")
        update_selection_via_textbox(None)
    pn.bind(parse_text_file, sids_upload_btn, watch=True)

    # Clear source id text box button
    sids_clear_btn = pn.widgets.Button(
        name="🗑️ Clear", width=button_width, disabled=False,
        description="Empties the text box"
    )

    def clear_text_box(value):
        sids_input.value = ""
        sids_upload_btn.clear()
    pn.bind(clear_text_box, value=sids_clear_btn, watch=True)

    # Source selection via embedding plot
    box_selector = streams.BoundsXY(source=bg_emb,
                                    bounds=(-1., -1., 1., 1.))

    def update_selection_via_plot(bounds: tuple[float, float, float, float]):
        sids = embedding.find_sids_in_box(
            box_bounds=bounds,
            x_dim=x_sel.value,
            y_dim=y_sel.value,
        )
        sids_upload_btn.clear()
        sids_input.value = "\n".join([str(s) for s in sids])
        sids_pipe.send(sids)
        update_plotted_sids(sids=sids)
    # TODO, WHY WATCH DOES NOT WORK WITH THIS ONE?
    bind_box_sel = pn.bind(update_selection_via_plot,
                           bounds=box_selector.param.bounds)

    def update_embedding_selection(data: list[int], x_dim: str, y_dim: str):
        coordinates = embedding.get_2d_embedding(
            sids=np.array(data), x_dim=x_dim, y_dim=y_dim)
        return hv.Points(
            coordinates, kdims=['x', 'y']
        ).opts(marker='star', size=5, alpha=0.25)

    sel_emb = hv.DynamicMap(update_embedding_selection,
                            streams={'data': sids_pipe} | emb_stream)

    # Update light curves and spectra
    resample_btn = pn.widgets.Button(
        name="🔄 Resample", width=button_width,
        description=(
            "Chooses 12 sources from the current selection "
            "and plots their light curves and spectra."
        )
    )
    pn.bind(update_plotted_sids, resample_btn.param.value, watch=True)

    # Light curve and spectra plots
    fold_check = pn.widgets.Checkbox(name='Fold light curves', value=False)

    def update_data_map(data: list[str],
                        plot_function: Callable,
                        folded: bool = False):
        n_plots = n_rows*n_cols
        plots = [plot_function(sid, folded=folded) for sid in data[:n_plots]]
        return hv.Layout(plots).cols(n_cols).opts(shared_axes=False)

    lc_dmap = hv.DynamicMap(
        partial(update_data_map, plot_function=plotter.plot_lightcurve),
        streams={'data': sids_pipe_plots, 'folded': fold_check.param.value},
    )
    xp_dmap = hv.DynamicMap(
        partial(update_data_map, plot_function=plotter.plot_spectra),
        streams=[sids_pipe_plots],
    )

    # Features distribution plot
    def update_features(data: list[int]):
        return plotter.plot_features(data)

    features_dmap = hv.DynamicMap(update_features, streams=[sids_pipe])

    # Sky map
    long, lat = embedding.get_galactic_coordinates()
    sky = gv.Points((long, lat), ['Longitude', 'Latitude'])
    bg_sky = hd.dynspread(hd.datashade(sky)).opts(
        projection=crs.Mollweide(), width=800, height=400)

    def update_sky_map(data: list[int]):
        data = [x for x in data if x is not None]
        if len(data) > 0:
            long, lat = embedding.get_galactic_coordinates(data)
        else:
            long, lat = [], []
        fg_sky = gv.Points((long, lat), ['Longitude', 'Latitude'])
        return fg_sky.opts(color='red', size=3, projection=crs.Mollweide())

    sky_dmap = hv.DynamicMap(update_sky_map, streams=[sids_pipe])

    # Dashboard
    tabs = pn.Tabs(
        ('Light curves', lc_dmap),
        ('Sampled spectra', xp_dmap),
        ('Sky map', hv.Overlay([
            bg_sky, sky_dmap,
        ]).collate()),
        ('Features', features_dmap),
        dynamic=True
    )
    bg_emb = datashade_embedding(bg_emb)
    return pn.Row(
        pn.Column(
            hv.Overlay([bg_emb, fg_emb, sel_emb]).collate(),
            pn.Row(x_sel, y_sel, class_sel),
            pn.Row(
                pn.Column(
                    pn.GridBox(
                        sids_copy_btn,
                        sids_submit_btn,
                        resample_btn,
                        sids_download_btn,
                        sids_clear_btn,
                        margin=0,
                        ncols=2,
                    ),
                    sids_upload_btn,
                ),
                sids_input,
            ),
        ),
        pn.Column(pn.Row(fold_check), tabs),
        pn.Column(pn.param.ParamFunction(bind_box_sel, loading_indicator=True)),
    )


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
    parser.add_argument('metadata_path', type=str)
    parser.add_argument('latent_dir', type=str)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    metadata_path = Path(args.metadata_path)
    latent_dir = Path(args.latent_dir)
    with open(data_dir / 'class_names.json', 'r') as f:
        class_metadata = json.load(f)
    if 'plotter' in pn.state.cache:
        plotter = pn.state.cache['plotter']
    else:
        pn.state.cache['plotter'] = plotter = DataLoader(data_dir, metadata_path)
    if 'emb' in pn.state.cache:
        emb = pn.state.cache['emb']
    else:
        pn.state.cache['emb'] = emb = Embedding(latent_dir, metadata_path, class_column='class')
    pn.extension(loading_indicator=True,
                 loading_spinner='dots',
                 global_loading_spinner=True)
    hv.extension('bokeh')
    gv.extension('bokeh')
    dashboard = build_panel(plotter, emb, class_metadata,
                            n_cols=3, n_rows=4)
    dashboard.servable(title='GaiaNet Embedding Explorer')
