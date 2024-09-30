from pathlib import Path
import argparse
from functools import partial
from typing import Callable
import polars as pl
import holoviews as hv
from holoviews import streams
import holoviews.operation.datashader as hd
import panel as pn
import datashader
import numpy as np

from data import DataLoader


colors = [
    'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink',
    'olive', 'cyan', 'magenta', 'yellow', 'teal', 'gold', 'indigo', 'lime',
    'navy', 'coral', 'salmon', 'darkgreen', 'orchid', 'sienna',
    'turquoise', 'maroon', 'darkblue'
]

short_names = {0: 'CEP', 1: 'ACV', 2: 'RR', 3: 'BCEP',
               4: 'BE', 5: 'SOLAR_LIKE', 6: 'YSO',
               7: 'CV', 8: 'DSCT|GDOR', 9: 'ECL', 10: 'WD',
               11: 'ELL', 12: 'LPV', 13: 'QSO', 14: 'RS',
               15: 'SPB', 16: 'SYST|ZAND'}


def build_panel(plotter: DataLoader,
                df_emb: pl.DataFrame,
                n_rows: int = 3,
                n_cols: int = 3):

    fold_check = pn.widgets.Checkbox(name='Fold light curves', value=False)
    resample_button = pn.widgets.Button(
        name="Reload source ids", button_type="success")
    sids_print = pn.widgets.TextAreaInput(
        value="", width=200, height=150, disabled=True
    )
    sids_copy_button = pn.widgets.Button(
        name="✂ Copy source ids to clipboard", button_type="success")
    sids_copy_button.js_on_click(
        args={"source": sids_print},
        code="navigator.clipboard.writeText(source.value);"
    )
    points = hv.Points(
        df_emb.select(['sourceid', 'embedding_0', 'embedding_1']).to_pandas(),
        kdims=['embedding_0', 'embedding_1']
    )
    labeled_meta = df_emb.filter(
        pl.col('label').ne(-1)
    ).select(['sourceid', 'embedding_0', 'embedding_1', 'label']).with_columns(
        pl.col('label').replace_strict(short_names).alias('class_name')
    )
    classes = hv.Points(
        labeled_meta.to_pandas(),
        kdims=['embedding_0', 'embedding_1'], vdims=['class_name']
    )
    aggregator = datashader.by('class_name', datashader.count())
    fg = hd.dynspread(
        hd.datashade(classes, aggregator=aggregator, color_key=colors),
        max_px=3, threshold=0.75, shape='square',
    )
    bg = hd.dynspread(
        hd.datashade(points, cmap="gray", cnorm="eq_hist"),
        max_px=3, threshold=0.75, shape='circle',
    )
    bg = bg.opts(width=650, height=500,
                 tools=['box_select'],
                 active_tools=['box_select', 'wheel_zoom'])
    fg = fg.opts(legend_position='right', fontsize={'legend': 8})

    initial_bounds = (-1., -1., 1., 1.)
    box_selector = streams.BoundsXY(source=points, bounds=initial_bounds)
    sids_holder = streams.Pipe(data=[])

    def in_bounds_expr(bounds: tuple[float, float, float, float],
                       xdim: str,
                       ydim: str):
        x_bounds = (pl.col(xdim) > bounds[0]) & (pl.col(xdim) < bounds[2])
        y_bounds = (pl.col(ydim) > bounds[1]) & (pl.col(ydim) < bounds[3])
        return x_bounds & y_bounds

    def update_selection_box(bounds: tuple[float, float, float, float],
                             value: bool = False,
                             n_rows: int = 4,
                             n_cols: int = 4):
        n_plots = n_rows*n_cols
        sids = df_emb.filter(
            in_bounds_expr(bounds, 'embedding_0', 'embedding_1')
        ).select('sourceid').to_series()
        if len(sids) > n_plots:
            sids = sids.sample(n_plots)
        sids = sids.to_list()
        sids_print.value = "\n".join([str(s) for s in sids])
        if len(sids) < n_plots:
            sids += [None]*(n_plots-len(sids))
        sids_holder.send(np.array(sids))
        return hv.Bounds(bounds)

    box_plot = hv.DynamicMap(partial(update_selection_box,
                                     n_rows=n_rows, n_cols=n_cols),
                             streams=[box_selector, resample_button.param.value])

    def update_data_map(data: list[int],
                        plot_function: Callable,
                        n_rows: int = 4,
                        n_cols: int = 4,
                        folded: bool = False):
        plots = [plot_function(sid, folded=folded) for sid in data]
        return hv.Layout(plots).cols(n_cols).opts(shared_axes=False)

    pn.extension()
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
        dynamic=True
    )
    return pn.Row(pn.Column(pn.pane.HoloViews(bg * fg * box_plot),
                            sids_print,
                            sids_copy_button),
                  pn.Column(pn.Row(fold_check, resample_button), tabs))


if __name__.startswith("bokeh"):
    parser = argparse.ArgumentParser(description='Panel')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('latent_dir', type=str)
    args = parser.parse_args()
    plotter = DataLoader(Path(args.data_dir))
    df_emb = pl.scan_parquet(
        Path(args.latent_dir) / '*.parquet'
    ).rename(
        {'source_id': 'sourceid'}
    ).select(['sourceid', 'embedding_0', 'embedding_1', 'label']).collect()
    dashboard = build_panel(plotter, df_emb, n_cols=3, n_rows=4)
    dashboard.servable()
