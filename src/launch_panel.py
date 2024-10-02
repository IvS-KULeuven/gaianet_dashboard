from pathlib import Path
import argparse
from functools import partial
from typing import Callable
import holoviews as hv
from holoviews import streams
import holoviews.operation.datashader as hd
import panel as pn
import datashader
import geoviews as gv
from cartopy import crs

from data_loader import DataLoader
from embedding import Embedding

colors = [
    'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink',
    'olive', 'cyan', 'magenta', 'yellow', 'teal', 'gold', 'indigo', 'lime',
    'navy', 'coral', 'salmon', 'darkgreen', 'orchid', 'sienna',
    'turquoise', 'maroon', 'darkblue'
]


def build_panel(plotter: DataLoader,
                embedding: Embedding,
                n_rows: int = 3,
                n_cols: int = 3):
    pn.extension()
    hv.extension('bokeh')
    gv.extension('bokeh')
    # GUI widgets
    fold_check = pn.widgets.Checkbox(name='Fold light curves', value=False)
    resample_button = pn.widgets.Button(
        name="Reload source ids", button_type="success")
    sids_print = pn.widgets.TextAreaInput(
        value="", width=250, height=150, disabled=True
    )
    sids_copy_button = pn.widgets.Button(
        name="✂ Copy source ids to clipboard", button_type="success")
    sids_copy_button.js_on_click(
        args={"source": sids_print},
        code="navigator.clipboard.writeText(source.value);"
    )
    # Sky map
    long, lat = embedding.get_galactic_coordinates()
    sky = gv.Points((long, lat), ['Longitude', 'Latitude'])
    bg_sky = hd.dynspread(hd.datashade(sky)).opts(
        projection=crs.Mollweide(), width=800, height=400)
    # Embedding plot
    embedding_unlabeled = hv.Points(
        embedding.get_embedding(),
        kdims=['x', 'y'], vdims=['sourceid']
    )
    embedding_labeled = hv.Points(
        embedding.get_labeled_embedding(),
        kdims=['x', 'y'], vdims=['label', 'sourceid']
    )
    aggregator = datashader.by('label', datashader.count())
    fg_emb = hd.dynspread(
        hd.datashade(embedding_labeled, aggregator=aggregator, color_key=colors),
        max_px=3, threshold=0.75, shape='square',
    ).opts(legend_position='right', fontsize={'legend': 8})

    bg_emb = hd.dynspread(
        hd.datashade(embedding_unlabeled, cmap="gray", cnorm="eq_hist"),
        max_px=3, threshold=0.75, shape='circle',
    ).opts(width=650, height=500,
           tools=['box_select'],
           active_tools=['box_select', 'wheel_zoom'])
    # Box selector
    box_selector = streams.BoundsXY(source=embedding_unlabeled,
                                    bounds=(-1., -1., 1., 1.))
    sids_holder = streams.Pipe(data=[])

    def update_selection_box(bounds: tuple[float, float, float, float],
                             value: bool = False,
                             n_rows: int = 4,
                             n_cols: int = 4):
        n_plots = n_rows*n_cols
        sids = embedding.find_sids_in_box(bounds, how_many=n_plots)
        sids_print.value = "\n".join([str(s) for s in sids])
        if len(sids) < n_plots:
            sids += [None]*(n_plots-len(sids))
        # Sending source ids to the pipe triggers
        # update on light curve, spectra and sky dynamic maps
        sids_holder.send(sids)
        return hv.Bounds(bounds)

    box_plot = hv.DynamicMap(
        partial(update_selection_box, n_rows=n_rows, n_cols=n_cols),
        streams=[box_selector, resample_button.param.value])

    def update_data_map(data: list[int],
                        plot_function: Callable,
                        n_rows: int = 4,
                        n_cols: int = 4,
                        folded: bool = False):
        plots = [plot_function(sid, folded=folded) for sid in data]
        return hv.Layout(plots).cols(n_cols).opts(shared_axes=False)

    def update_sky_map(data: list[int]):
        data = [x for x in data if x is not None]
        if len(data) > 0:
            long, lat = embedding.get_galactic_coordinates(data)
        else:
            long, lat = [], []
        fg_sky = gv.Points((long, lat), ['Longitude', 'Latitude'])
        return fg_sky.opts(color='red', size=10, projection=crs.Mollweide())

    update_lc = partial(update_data_map,
                        plot_function=plotter.plot_lightcurve,
                        n_cols=n_cols, n_rows=n_rows)
    update_xp = partial(update_data_map,
                        plot_function=plotter.plot_spectra,
                        n_cols=n_cols, n_rows=n_rows)
    # stats_dmap = hv.DynamicMap(update_stats, streams=[box])
    lc_streams = {'data': sids_holder, 'folded': fold_check.param.value}
    tabs = pn.Tabs(
        ('Light curves', hv.DynamicMap(update_lc, streams=lc_streams)),
        ('Sampled spectra', hv.DynamicMap(update_xp, streams=[sids_holder])),
        ('Sky map', bg_sky*hv.DynamicMap(update_sky_map, streams=[sids_holder])),
        dynamic=True
    )
    return pn.Row(pn.Column(pn.pane.HoloViews(bg_emb * fg_emb * box_plot),
                            sids_print,
                            sids_copy_button),
                  pn.Column(pn.Row(fold_check, resample_button), tabs))


if __name__.startswith("bokeh"):
    parser = argparse.ArgumentParser(description='Panel')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('latent_dir', type=str)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    latent_dir = Path(args.latent_dir)
    plotter = DataLoader(data_dir)
    emb = Embedding(latent_dir)
    dashboard = build_panel(plotter, emb, n_cols=3, n_rows=4)
    dashboard.servable(title='GaiaNet embedding explorer')
