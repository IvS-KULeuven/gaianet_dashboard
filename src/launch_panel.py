import io
from pathlib import Path
import argparse
import json
import time
from functools import partial
from typing import Callable
import logging
import numpy as np
import datashader as ds
import holoviews as hv
from holoviews import streams
from holoviews.selection import link_selections
import holoviews.operation.datashader as hd
import panel as pn
import geoviews as gv
from cartopy import crs

from data_loader import DataLoaderSQLite
from metadata import MetadataHandler
from plots import lc_layout, dmdt_layout, xp_layout

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


def build_panel(plotter: DataLoaderSQLite,
                embedding: MetadataHandler,
                class_metadata: dict,
                data_dir: Path,
                n_rows: int = 3,
                n_cols: int = 3):
    n_plots = n_rows*n_cols
    button_width = 100

    # Pipes for passing source ids
    sids_pipe = streams.Pipe(data=[])
    ls = link_selections.instance(unselected_alpha=1.)

    # Embedding plot
    def plot_embedding(x_dim: str,
                       y_dim: str,
                       class_name: str | None = None,
                       sids: np.ndarray | None = None,
                       **plot_kwargs):
        kdims = [x_dim, y_dim]
        if class_name is None and sids is None:
            return hv.Points(embedding.metadata_hv, kdims)
        return hv.Points(
            embedding.filter_embedding(sids=sids, class_name=class_name), kdims
        ).opts(**plot_kwargs)

    def plot_labeled_embedding(x_dim, y_dim,
                               class_name=None,
                               sids=None):
        sel_emb = embedding.get_2d_embedding(
            sids=None, class_name=None, x_dim=x_dim, y_dim=y_dim,
        )
        mask = sel_emb['label'] != 'UNKNOWN'
        return hv.Points(
            {'x': sel_emb['x'][mask],
             'y': sel_emb['y'][mask],
             'label': sel_emb['label'][mask]
             }, kdims=['x', 'y'], vdims=['label'],
        )

    emb_columns = emb.emb_columns
    emb_select = partial(pn.widgets.Select, width=150, options=emb_columns)
    x_sel = emb_select(value=emb_columns[0])
    y_sel = emb_select(value=emb_columns[1])
    emb_stream = {'x_dim': x_sel.param.value, 'y_dim': y_sel.param.value}
    bg_emb = hv.DynamicMap(plot_embedding, streams=emb_stream)
    #bg_emb2 = hv.DynamicMap(plot_labeled_embedding, streams=emb_stream)

    #aggregator = ds.by('label', ds.count())
    #bg_emb2_dyn = hd.dynspread(
    #    hd.datashade(bg_emb2, aggregator=aggregator, color_key=colors),
    #    max_px=3, threshold=0.75, shape='square',
    #).opts(legend_position='right', fontsize={'legend': 8})

    class_sel = pn.widgets.Select(groups=class_metadata, value='none')

    # Source selection via textbox
    sids_input = pn.widgets.TextAreaInput(
        value="", width=320, height=180, disabled=False, max_length=500000,
        placeholder=(
            "Enter one Gaia DR3 source id per line, "
            "e.g\n2190530735119392128\n5874749936451323264\n"
            "...\nor upload them in a plain text file. "
            "Then press submit to highlight them in the embedding. "
        ), styles={'font-size': '12px'}
    )
    sids_submit_btn = pn.widgets.Button(
        name="✅ Submit", width=button_width)

    def update_selection_via_textbox(value):
        tinit = time.time()
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
        logger.info(f"Sending to sids_pipe: {time.time()-tinit:0.4f}")
        # update_plotted_sids(sids=sids)

    # Copy selection to clipboard
    sids_copy_btn = pn.widgets.Button(
        name="📋 Copy", width=button_width,
        description="Copies the content of the text box to your clipboard"
    )
    sids_copy_btn.js_on_click(
        args={"source": sids_input},
        code="navigator.clipboard.writeText(source.value);"
    )

    
    def disable_download(event):
        sids = sids_input.value
        if sids == '':
            sids_download_btn.disabled = True
            sids_clear_btn.disabled = True
        else:
            sids_download_btn.disabled = False
            sids_clear_btn.disabled = False
    # pn.bind(disable_download, sids_input.param.value, watch=True)

    # Upload CSV with selection
    sids_upload_btn = pn.widgets.FileInput(
        name="📤 Upload", width=3*button_width, sizing_mode="fixed",
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
                                    bounds=(-0.1, -0.1, 0.1, 0.1))

    def update_selection_via_plot(bounds: tuple[float, float, float, float]):
        tinit = time.time()
        sids = embedding.find_sids_in_box(
            box_bounds=bounds,
            x_dim=x_sel.value,
            y_dim=y_sel.value,
        )
        sids_upload_btn.clear()
        sids_input.value = "\n".join([str(s) for s in sids])
        sids_pipe.send(sids)
        logger.info(f"Sending to sids_pipe: {time.time()-tinit:0.4f}")
        update_plotted_sids(sids=sids)
    # TODO, WHY WATCH DOES NOT WORK WITH THIS ONE?
    #bind_box_sel = pn.bind(update_selection_via_plot,
    #                       bounds=box_selector.param.bounds)

    def update_embedding_selection(data: list[int], x_dim: str, y_dim: str):
        tinit = time.time()
        coordinates = embedding.get_2d_embedding(
            sids=np.array(data), x_dim=x_dim, y_dim=y_dim)
        fg_emb = hv.Points(
            coordinates, kdims=['x', 'y']
        ).opts(marker='star', size=5, alpha=0.25)
        logger.info(f"Update embedding: {time.time()-tinit:0.4f}")
        return fg_emb

    #sel_emb = hv.DynamicMap(update_embedding_selection,
    #                        streams={'data': sids_pipe} | emb_stream)
    class UserData:
        def __init__(self):
            self.selected_data = None
            self.user_selected_class = None
            self.last_expr = None
            self.sids = [None]*n_plots
            self.labels = [None]*n_plots
            self.lcs = [None]*n_plots
            self.xps = [None]*n_plots
            self.dmdts = [None]*n_plots
            self.freqs = [None]*n_plots
            self.update_trigger = pn.widgets.Button(visible=False)
            self.resample_trigger = pn.widgets.Button(visible=False)

        def update_selected_class(self, data):
            self.user_selected_class = data

        def print_select_sids(self):
            if self.selected_data is not None:
                sids = self.selected_data['source_id'].to_numpy().astype('str')
                sids_txt = '\n'.join(sids.tolist())
                return io.StringIO(sids_txt)

        def update_selection_via_plot(self, expr):
            print("update_selection", expr, ls.selection_expr, self.last_expr)
            expr = ls.selection_expr
            if expr is None:
                return
            if not str(expr) == str(self.last_expr):
                tinit = time.time()
                self.selected_data = embedding.metadata_hv.select(expr).data
                self.last_expr = expr
                logger.info(f"Update selection via box: {time.time()-tinit:0.4f}")
                self.resample()
                self.update_trigger.clicks += 1

        def resample(self, *args, **kwargs):
            print("resample", args, kwargs)
            if self.selected_data is None:
                return
            tinit = time.time()
            sids = self.selected_data['source_id'].to_numpy()
            freqs = self.selected_data['NUFFT_best_frequency'].to_numpy()
            labels = self.selected_data['class'].to_numpy()
            if len(sids) > n_plots:
                perm = np.random.permutation(len(sids))[:n_plots]
                sids = sids[perm]
                freqs = freqs[perm]
                labels = labels[perm]
            sids = sids.tolist()
            freqs = freqs.tolist()
            labels = labels.tolist()
            self.sids = sids
            self.freqs = freqs
            self.labels = labels
            self.lcs, self.xps, self.dmdts = [], [], []
            for sid in sids:
                lc, xp, dmdt = plotter.retrieve_data(sid)
                self.lcs.append(lc)
                self.xps.append(xp)
                self.dmdts.append(dmdt)
            if len(sids) < n_plots:
                to_add = n_plots - len(sids)
                self.sids += [None]*to_add
                self.lcs += [None]*to_add
                self.xps += [None]*to_add
                self.labels += [None]*to_add
                self.dmdts += [None]*to_add
                self.freqs += [None]*to_add
            logger.info(f"Update lc and xp: {time.time()-tinit:0.4f}")
            self.resample_trigger.clicks += 1

    user_data = UserData()

    # Download selection as CSV
    sids_download_btn = pn.widgets.FileDownload(
        callback=user_data.print_select_sids, filename='sids.txt',
        label="⬇️  Download", width=button_width,
        description=(
            "Downloads the content of the text box as a sids.txt file"
        )
    )

    def plot_class(x_dim, y_dim, name='none'):
        if name == 'none':
            return hv.Points([])
        return plot_embedding(
            class_name=name, x_dim=x_dim, y_dim=y_dim, sids=None, alpha=0.5, color='blue',
        )
    sel_class_emb = hv.DynamicMap(
        plot_class,
        streams=emb_stream | {'name': class_sel.param.value}
    )

    # Light curve and spectra plots
    fold_check = pn.widgets.Checkbox(name='Fold light curves', value=False)

    def update_lc_map(trigger, fold, highlight_label):
        tinit = time.time()
        sids = user_data.sids
        lcs = user_data.lcs
        labels = user_data.labels
        if fold:
            freqs = user_data.freqs
        else:
            freqs = [None]*n_plots
        layout = lc_layout(data=lcs, sids=sids, labels=labels, frequencies=freqs, highlight_label=highlight_label, n_cols=3)
        logger.info(f"Update lc data map: {time.time()-tinit:0.4f}")
        return layout

    def update_xp_map(trigger, highlight_label):
        tinit = time.time()
        sids = user_data.sids
        xps = user_data.xps
        labels = user_data.labels
        freqs = [None]*n_plots
        layout = xp_layout(data=xps, sids=sids, labels=labels, frequencies=freqs, highlight_label=highlight_label, n_cols=3)
        logger.info(f"Update xp data map: {time.time()-tinit:0.4f}")
        return layout

    def update_dmdt_map(trigger, highlight_label):
        tinit = time.time()
        sids = user_data.sids
        dmdts = user_data.dmdts
        labels = user_data.labels
        freqs = [None]*n_plots
        layout = dmdt_layout(data=dmdts, sids=sids, labels=labels, frequencies=freqs, highlight_label=highlight_label, n_cols=3)
        logger.info(f"Update dmdt data map: {time.time()-tinit:0.4f}")
        return layout

    # Update light curves and spectra
    resample_btn = pn.widgets.Button(
        name="🔄 Resample", width=button_width,
        description=(
            "Chooses 12 sources from the current selection "
            "and plots their light curves and spectra."
        )
    )
    pn.bind(user_data.resample, resample_btn.param.value, watch=True)
    pn.bind(user_data.update_selection_via_plot, box_selector.param.bounds, watch=True)
    lc_dmap = hv.DynamicMap(
        pn.bind(update_lc_map,
                trigger=user_data.resample_trigger.param.clicks,
                fold=fold_check.param.value,
                highlight_label=class_sel.param.value)
    )
    xp_dmap = hv.DynamicMap(
        pn.bind(update_xp_map,
                trigger=user_data.resample_trigger.param.clicks,
                highlight_label=class_sel.param.value)
    )
    dmdt_dmap = hv.DynamicMap(
        pn.bind(update_dmdt_map,
                trigger=user_data.resample_trigger.param.clicks,
                highlight_label=class_sel.param.value)
    )

    def update_feature_map(trigger, selected_class):
        tinit = time.time()
        class_selection = None
        if selected_class != 'none':
            # USER DATA SHOULD UPDATE THIS ONCE
            class_selection = embedding.filter_embedding(class_name=selected_class, sids=None)
        plots = []
        for col in ['magnitude_mean', 'magnitude_std', 'bp_rp', 'ruwe', 'NUFFT_best_frequency']:
            data = []
            if user_data.selected_data is not None:
                data = user_data.selected_data[col]
            #if 'frequency' in col:
            #    data = np.log10(data)
            plot_u = hv.Distribution((data), kdims=[col]).opts(color='red', framewise=True)
            data = []
            if class_selection is not None:
                data = class_selection[col]
            plot_l = hv.Distribution(data, kdims=[col]).opts(color='blue', framewise=True)
            plots.append(hv.Overlay([plot_u, plot_l]).opts(shared_axes=True, width=350, height=200))

        bar_data = [('', 0)]
        if user_data.selected_data is not None:
            data = user_data.selected_data['class']
            labels, counts = np.unique(data.dropna().to_numpy(), return_counts=True)
            idx = np.argsort(counts)[::-1]
            labels, counts = labels[idx], counts[idx]
            if len(labels) > 5:
                labels = labels[:5]
                counts = counts[:5]
            bar_data = [(str(label), count) for label, count in zip(labels, counts)]
        plots.append(
            hv.Bars(
                bar_data, kdims=['Class'], vdims=['Count']
            ).opts(width=350, height=200, framewise=True)
        )
        logger.info(f"Building feature plot: {time.time()-tinit:0.4f}")
        return hv.Layout(plots).cols(2).opts(shared_axes=False)
    feature_dmap = hv.DynamicMap(
        pn.bind(update_feature_map,
                trigger=user_data.update_trigger.param.clicks,
                selected_class=class_sel.param.value)
    )

    # Sky map
    sky = gv.Points(embedding.metadata_hv, kdims=['longitude', 'latitude'])
    bg_sky = hd.dynspread(hd.datashade(sky, cmap="gray", cnorm='eq_hist')).opts(
        projection=crs.Mollweide(), width=800, height=400)

    def update_sky_map(trigger, selected_class):
        class_selection = None
        if selected_class != 'none':
            class_selection = embedding.filter_embedding(class_name=selected_class, sids=None)
        tinit = time.time()
        kdims = ['longitude', 'latitude']
        data = []
        if user_data.selected_data is not None:
            data = user_data.selected_data
        plot_u = gv.Points(data, kdims=kdims).opts(color='red', size=3, alpha=0.5, projection=crs.Mollweide())
        data = []
        if class_selection is not None:
            data = class_selection
        plot_l = gv.Points(data, kdims=kdims).opts(color='blue', size=3, alpha=0.5, projection=crs.Mollweide())
        plot = hv.Overlay([plot_u, plot_l])
        logger.info(f"Update sky map: {time.time()-tinit:0.4f}")
        return plot

    sky_dmap = hv.DynamicMap(
        pn.bind(update_sky_map,
                trigger=user_data.update_trigger.param.clicks,
                selected_class=class_sel.param.value)
    )

    # Dashboard
    tabs = pn.Tabs(
        ('Light curves', lc_dmap),
        ('Sampled spectra', xp_dmap),
        ('DMDT', dmdt_dmap),
        ('Sky map', hv.Overlay([
            bg_sky * sky_dmap,
        ]).collate()),
        ('Features', feature_dmap),
        dynamic=True
    )
    bg_emb = datashade_embedding(bg_emb)
    return pn.Row(
        pn.Column(
            hv.Overlay([ls(bg_emb), sel_class_emb]).collate(),
            pn.Row(x_sel, y_sel, class_sel),
            pn.Row(
                pn.Column(
                    pn.GridBox(
                        sids_copy_btn,
                        sids_submit_btn,
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
        pn.Column(pn.Row(fold_check, resample_btn), tabs),
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
    parser.add_argument('latent_path', type=str)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    latent_path = Path(args.latent_path)
    # More details about the classes at:
    # https://www.aanda.org/articles/aa/full_html/2023/06/aa45591-22/aa45591-22.html
    with open(data_dir / 'class_names.json', 'r') as f:
        class_metadata = json.load(f)
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
            data_dir / 'metadata' / 'meta.csv',
            data_dir / 'metadata' / 'labels.csv',
        )
    pn.extension(loading_indicator=True,
                 loading_spinner='dots',
                 global_loading_spinner=True)
    hv.extension('bokeh')
    gv.extension('bokeh')
    dashboard = build_panel(plotter, emb, class_metadata, data_dir,
                            n_cols=3, n_rows=3)
    dashboard.servable(title='GaiaNet Embedding Explorer')
