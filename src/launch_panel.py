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


def build_panel(data_path, df_meta, n_rows=3, n_cols=3):

    fold_check = pn.widgets.Checkbox(name='Fold light curves', value=False)
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
        df_meta.select(['sourceid', 'embedding_0', 'embedding_1']).to_pandas(),
        kdims=['embedding_0', 'embedding_1']
    )
    labeled_meta = df_meta.filter(
        pl.col('label').ne(-1)
    ).select(['sourceid', 'embedding_0', 'embedding_1', 'label']).with_columns(
        pl.col('label').replace_strict(short_names).alias('class_name')
    )
    classes = hv.Points(
        labeled_meta.to_pandas(),
        kdims=['embedding_0', 'embedding_1'], vdims=['class_name']
    )
    fg = hd.dynspread(
        hd.datashade(classes,
                     aggregator=datashader.by('class_name', datashader.count()),
                     color_key=colors),
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
    box_plot = hv.DynamicMap(lambda bounds: hv.Bounds(bounds),
                             streams=[box_selector])

    def in_bounds_expr(bounds: tuple[float, float, float, float],
                       xdim: str,
                       ydim: str):
        x_bounds = (pl.col(xdim) > bounds[0]) & (pl.col(xdim) < bounds[2])
        y_bounds = (pl.col(ydim) > bounds[1]) & (pl.col(ydim) < bounds[3])
        return x_bounds & y_bounds

    def update_data_map(bounds: tuple[float, float, float, float],
                        plot_function: Callable,
                        n_rows: int = 4,
                        n_cols: int = 4):
        n_plots = n_rows*n_cols
        sids = df_meta.filter(
            in_bounds_expr(bounds, 'embedding_0', 'embedding_1')
        ).select('sourceid').to_series()
        if len(sids) > n_plots:
            sids = sids.head(n_plots)
        sids_print.value = "\n".join([str(s) for s in sids])
        plots = [plot_function(sid, folded=fold_check.value) for sid in sids]
        if len(plots) < n_plots:
            plots += [plot_function(None) for _ in range(n_plots-len(plots))]
        return hv.Layout(plots).cols(n_cols).opts(shared_axes=False)

    pn.extension()
    plotter = DataLoader(data_path)
    update_lc = partial(update_data_map,
                        plot_function=plotter.plot_lightcurve,
                        n_cols=n_cols, n_rows=n_rows)
    update_xp = partial(update_data_map,
                        plot_function=plotter.plot_spectra,
                        n_cols=n_cols, n_rows=n_rows)
    # stats_dmap = hv.DynamicMap(update_stats, streams=[box])
    tabs = pn.Tabs(
        ('Light curves', hv.DynamicMap(update_lc, streams=[box_selector])),
        ('Sampled spectra', hv.DynamicMap(update_xp, streams=[box_selector]))
    )
    return pn.Row(pn.Column(pn.pane.HoloViews(bg * fg * box_plot),
                            sids_print,
                            sids_copy_button),
                  pn.Column(fold_check, tabs))

# if __name__.startswith("bokeh"):
#    dashboard = build_panel()
#    dashboard.servable()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Panel')
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    root_path = Path(args.path)
    data_path = root_path / 'data' / 'DR3_40obs_20mag_public_with_spectra'
    emb_path = root_path / 'results' / 'latent_space' / '2'
    meta_path = root_path / 'data' / 'training_sets' / 'DR3_40obs_20mag_public_with_spectra.parquet'

    df_embedding = pl.scan_parquet(Path(emb_path) / '*.parquet').rename({'source_id': 'sourceid'})
    df_meta = pl.scan_parquet(meta_path)
    df_meta = df_meta.join(df_embedding, on='sourceid').collect()

    dashboard = build_panel(data_path, df_meta, n_cols=3, n_rows=4)
    dashboard.show(port=5007)
