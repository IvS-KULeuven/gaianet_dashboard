from pathlib import Path
import argparse
from functools import partial
from typing import Callable
import logging
import holoviews as hv
from holoviews import streams
import holoviews.operation.datashader as hd
import panel as pn
import datashader
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
                n_rows: int = 3,
                n_cols: int = 3):
    pn.extension()
    hv.extension('bokeh')
    gv.extension('bokeh')
    n_plots = n_rows*n_cols
    button_width = 100

    # Pipes for passing source ids
    sids_pipe = streams.Pipe(data=[])
    sids_pipe_plots = streams.Pipe(data=[])

    # Embedding plot
    def plot_embedding(x_col, y_col):
        embedding.set_plot_columns(x_col, y_col)
        return hv.Points(
            embedding.get_embedding(),
            kdims=['x', 'y'], vdims=['sourceid']
        ).opts(framewise=True)
        return embedding_unlabeled
        embedding_labeled = hv.Points(
            embedding.get_labeled_embedding(),
            kdims=['x', 'y'], vdims=['label', 'sourceid']
        )
        aggregator = datashader.by('label', datashader.count())
        fg_emb = hd.dynspread(
            hd.datashade(embedding_labeled,
                         aggregator=aggregator,
                         color_key=colors),
            max_px=3, threshold=0.75, shape='square',
        ).opts(legend_position='right', fontsize={'legend': 8})

        bg_emb = hd.dynspread(
            hd.datashade(embedding_unlabeled,
                         cmap="gray",
                         cnorm="eq_hist"),
            max_px=3, threshold=0.75, shape='circle',
        ).opts(width=650, height=500,
               tools=['box_select'],
               active_tools=['box_select', 'wheel_zoom'])
        return bg_emb
    x_sel = pn.widgets.Select(options=emb.emb_columns, value=emb.emb_x, width=150)
    y_sel = pn.widgets.Select(options=emb.emb_columns, value=emb.emb_y, width=150)
    emb_stream = {'x_col': x_sel.param.value, 'y_col': y_sel.param.value}
    bg_emb = hv.DynamicMap(plot_embedding, streams=emb_stream).opts(framewise=True)

    class_sel = pn.widgets.Select(options=['none'] + emb.available_classes,
                                  value='none')

    def plot_class(x_col, y_col, name='none'):
        if name == 'none':
            return hv.Points([])
        return hv.Points(
            embedding.get_class_embedding(class_name=name),
            kdims=['x', 'y'], vdims=['sourceid']
        ).opts(alpha=0.5, framewise=True)
    fg_emb = hv.DynamicMap(
        plot_class,
        streams=emb_stream | {'name': class_sel.param.value}
    ).opts(framewise=True)

    # Source selection via textbox
    sids_input = pn.widgets.TextAreaInput(
        value="", width=300, height=200, disabled=False, max_length=500000,
        placeholder=(
            "Enter your source ids line by line, "
            "e.g\n2190530735119392128\n5874749936451323264\n"
            "...\nor upload them in a plain text file. "
            "Then press submit to highlight them in the embedding. "
            "Press resample to change the plotted light curves."
        ), styles={'font-size': '12px'}
    )
    sids_submit_btn = pn.widgets.Button(
        name="✅ Submit", width=button_width)

    def update_plotted_sids(value=None):
        sids = embedding.selection
        if len(sids) > n_plots:
            sids = sids.sample(n_plots)
        sids = sids.to_list()
        if len(sids) < n_plots:
            sids += [None]*(n_plots-len(sids))
        sids_pipe_plots.send(sids)

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
        update_plotted_sids()
    bind_text_sel = pn.bind(update_selection_via_textbox,
                            value=sids_submit_btn)

    # Clear source id text box button
    sids_clear_btn = pn.widgets.Button(
        name="🗑️ Clear", width=button_width,
    )

    def clear_text_box(value):
        sids_input.value = ""
    bind_text_clear = pn.bind(clear_text_box, value=sids_clear_btn)

    # Copy selection to clipboard
    sids_copy_btn = pn.widgets.Button(
        name="📋 Copy", width=button_width)
    sids_copy_btn.js_on_click(
        args={"source": sids_input},
        code="navigator.clipboard.writeText(source.value);"
    )

    # Download selection as CSV
    # TODO
    sids_download_btn = pn.widgets.Button(
        name="⬇️  Download", width=button_width)
    # Upload CSV with selection
    # TODO
    sids_upload_btn = pn.widgets.Button(
        name="📤 Upload", width=button_width)

    # Source selection via embedding plot
    box_selector = streams.BoundsXY(source=bg_emb,
                                    bounds=(-1., -1., 1., 1.))

    def update_selection_via_plot(bounds: tuple[float, float, float, float]):
        sids = embedding.find_sids_in_box(bounds)
        sids_input.value = "\n".join([str(s) for s in sids])
        sids_pipe.send(sids)
        update_plotted_sids()
    bind_box_sel = pn.bind(update_selection_via_plot,
                           bounds=box_selector.param.bounds)

    # THIS ALSO HAS TO BE UPDATE BY SELECTORS
    def update_embedding_selection(data: list[int], x_col: str, y_col: str):
        coordinates = embedding.get_embedding_for_selection(data)
        return hv.Points(
            coordinates, kdims=['x', 'y']
        ).opts(marker='star', size=5, alpha=0.25)

    sel_emb = hv.DynamicMap(update_embedding_selection,
                            streams={'data': sids_pipe} | emb_stream).opts(framewise=True)

    # Update light curves and spectra
    resample_btn = pn.widgets.Button(
        name="🔄 Resample", width=button_width)
    bind_reload = pn.bind(update_plotted_sids, resample_btn.param.value)

    # Light curve and spectra plots
    fold_check = pn.widgets.Checkbox(name='Fold light curves', value=False)

    def update_data_map(data: list[int],
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
    bg_emb = datashade_embedding(bg_emb).opts(framewise=True)
    return pn.Row(
        pn.Column(
            hv.Overlay([bg_emb, fg_emb, sel_emb]).opts(framewise=True),
            pn.Row(x_sel, y_sel, class_sel),
            pn.Row(
                sids_input,
                pn.GridBox(
                    sids_upload_btn,
                    sids_copy_btn,
                    sids_submit_btn,
                    resample_btn,
                    sids_download_btn,
                    sids_clear_btn,
                    margin=0,
                    ncols=2,
                ),
            ),
        ),
        pn.Column(pn.Row(fold_check), tabs),
        pn.Column(bind_box_sel, bind_text_sel, bind_text_clear, bind_reload),
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
    emb = Embedding(latent_dir, metadata_path, class_column='class')
    plotter = DataLoader(data_dir, metadata_path)
    dashboard = build_panel(plotter, emb, n_cols=3, n_rows=4)
    dashboard.servable(title='GaiaNet Embedding Explorer')
