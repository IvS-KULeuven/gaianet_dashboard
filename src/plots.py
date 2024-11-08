from typing import Callable
from itertools import pairwise
from functools import partial
import numpy as np
import holoviews as hv
import holoviews.operation.datashader as hd
import geoviews as gv
from cartopy import crs


def labeled_bgcolor(plot, _):
    plot.handles['plot'].border_fill_color = 'lightblue'


def regular_bgcolor(plot, _):
    plot.handles['plot'].border_fill_color = None


def format_title(sid: int | str | None,
                 label: str | None,
                 frequency: float | None = None
                 ) -> str:
    if sid is None:
        return ''
    if isinstance(sid, int):
        sid = str(sid)
    title = sid
    if label is not None:
        title += f' {label}'
    if frequency is not None:
        title += f' f={frequency:5.5g}'
    return title


def plot_lightcurve(data: np.ndarray | None,
                    sid: str | None,
                    label: str | None,
                    frequency: float | None = None,
                    width: int = 285,
                    height: int = 200,
                    plot_errors: bool = True,
                    fold_with_double_period: bool = True,
                    ):
    kdims = ['Time']
    title = format_title(sid, label, frequency)
    if data is None:
        time, mag, err = [], [], []
    else:
        time, mag, err = data
        if frequency is not None:
            P = 1.0/frequency
            if fold_with_double_period:
                P = 2*P
            time = np.mod(time, P)/P
            if fold_with_double_period:
                time = 2*time
            kdims = ['Phase']
    dots = hv.Scatter(
        (time, mag), kdims=kdims, vdims=['Magnitude']
    ).opts(color='g', framewise=True, invert_yaxis=True,
           width=width, height=height)
    plots = [dots]
    if plot_errors:
        bars = hv.ErrorBars(
            (time, mag, err), kdims=kdims, vdims=['Magnitude', 'Error']
        ).opts(color='g', lower_head=None, upper_head=None)
        plots.append(bars)
    return hv.Overlay(
        plots
    ).opts(shared_axes=True, title=title, fontsize={'title': 8})


def make_layout(plot_function: Callable,
                data: list,
                sids: list,
                labels: list,
                frequencies: list,
                highlight_label: str,
                n_cols: int):
    plots = []
    for item, sid, label, frequency in zip(data, sids, labels, frequencies):
        hooks = [regular_bgcolor]
        if label is not None and highlight_label != 'none':
            if label == highlight_label:
                hooks = [labeled_bgcolor]
        plots.append(
            plot_function(
                data=item, sid=sid, label=label, frequency=frequency
            ).opts(hooks=hooks)
        )
    return hv.Layout(
        plots
    ).cols(n_cols).opts(shared_axes=False, framewise=True)


def plot_spectra(data: np.ndarray | None,
                 sid: str | None,
                 label: str | None,
                 width: int = 285,
                 height: int = 200,
                 **kwargs,
                 ):
    title = format_title(sid, label)
    if data is None:
        pw, bp, rp, bp_err, rp_err = [], [], [], [], []
    else:
        pw = np.linspace(0, 60, 55)
        bp, rp = data[0, 0], data[1, 0]
        bp_err, rp_err = data[0, 1], data[1, 1]
    c_bp = hv.Curve(
        (pw, bp), kdims=['Pseudo frequency'], vdims=['Flux']
    ).opts(color='b', framewise=True, width=width, height=height)
    c_bp_e = hv.Spread(
        (pw, bp, bp_err), kdims=['Pseudo frequency'], vdims=['Flux', 'Err']
    ).opts(fill_color='b', line_width=0, fill_alpha=0.5, framewise=True)
    c_rp = hv.Curve(
        (pw, rp), kdims=['Pseudo frequency'], vdims=['Flux']
    ).opts(color='r', framewise=True)
    c_rp_e = hv.Spread(
        (pw, rp, rp_err), kdims=['Pseudo frequency'], vdims=['Flux', 'Err']
    ).opts(fill_color='r', line_width=0, fill_alpha=0.5, framewise=True)
    return hv.Overlay(
        [c_bp, c_bp_e, c_rp, c_rp_e]
    ).opts(shared_axes=True, title=title, fontsize={'title': 8})


time_bins = [0.07, 0.1, 0.5, 20, 50]  # Right bin is open
magnitude_bins = np.linspace(-3, 3, 22)
xticks, yticks = [], []
xticks = [f'({b1},{b2})' for b1, b2 in pairwise(time_bins)]
xticks += [f'({time_bins[-1]},inf)']
yticks += [f'(-inf,{magnitude_bins[0]:3.2f})']
yticks += [f'({b1:3.2f},{b2:3.2f})' for b1, b2 in pairwise(magnitude_bins)]
yticks += [f'({magnitude_bins[-1]:3.2f},inf)']


def plot_dmdt(data: np.ndarray | None,
              sid: str | None,
              label: str | None,
              width: int = 285,
              height: int = 250,
              **kwargs,
              ):
    title = format_title(sid, label)
    kdims = ['dt bin', 'dm bin']
    if data is not None:
        plot_data = (xticks, yticks, data)
    else:
        plot_data = (xticks, yticks, np.zeros(shape=(23, 5)))
    plot = hv.HeatMap(plot_data, kdims=kdims)
    return plot.opts(
        title=title,  tools=['hover'], xrotation=0,
        fontsize={'title': 8, 'labels': 8, 'xticks': 6, 'yticks': 6},
        cmap='blues', clim=(0, 1), framewise=True, width=width, height=height
    )


lc_layout = partial(make_layout, plot_function=plot_lightcurve)
xp_layout = partial(make_layout, plot_function=plot_spectra)
dmdt_layout = partial(make_layout, plot_function=plot_dmdt)


# TODO: MAKE DYNAMIC MAP SCALE THE AXES
def datashade_embedding(emb_plot,
                        cnorm: str = "eq_hist",
                        cmap: str = "gray",
                        width: int = 650,
                        height: int = 500,
                        bound: float = 3.5):
    plot = hd.datashade(emb_plot, cmap=cmap, cnorm=cnorm)
    plot = hd.dynspread(plot, max_px=3, threshold=0.75, shape='circle')
    return plot.opts(
        width=width, height=height, xlim=(-bound, bound), ylim=(-bound, bound),
        tools=['box_select'], active_tools=['box_select', 'wheel_zoom']
    )


def plot_sky(coordinates: np.ndarray | list,
             color: str = 'gray'):
    kdims = ['longitude', 'latitude']
    return gv.Points(coordinates, kdims=kdims).opts(
        color=color, size=3, alpha=0.5, projection=crs.Mollweide()
    )


def datashade_skymap(sky_plot,
                     cnorm: str = "eq_hist",
                     cmap: str = "gray",
                     width: int = 800,
                     height: int = 400):
    plot = hd.datashade(sky_plot, cmap=cmap, cnorm=cnorm)
    plot = hd.dynspread(plot)
    return plot.opts(
        projection=crs.Mollweide(), width=width, height=height
    )


def label_histogram(data: np.ndarray):
    bar_data = [('', 0)]
    data = data[data != np.array(None)]
    labels, counts = np.unique(data, return_counts=True)
    if len(counts) > 0:
        idx = np.argsort(counts)[::-1]
        labels, counts = labels[idx], counts[idx]
        if len(labels) > 5:
            labels = labels[:5]
            counts = counts[:5]
        bar_data = [(str(label), count) for label, count in zip(labels, counts)]
    return bar_data


def plot_features(user_data):
    plots = []
    pretty_name = {'magnitude_mean': 'Mean magnitude (g)',
                   'magnitude_std': 'StdDev magnitude (g)',
                   'bp_rp': 'Color (Mean Bp - Mean Rp)',
                   'NUFFT_best_frequency': 'Log dominant period',
                   'period': 'Dominant period [days]'}
    for col in ['magnitude_mean', 'magnitude_std', 'bp_rp', 'NUFFT_best_frequency']:
        fplots = []
        for selection, color in zip(
            [user_data.selected_emb, user_data.selected_cu7,
             user_data.selected_sos, user_data.selected_vari],
            ['red', 'blue', 'green', 'green']
        ):
            data = [0, 0]
            if selection.has_data:
                data = selection.features[col]
            col_name = pretty_name[col]
            fplots.append(
                hv.Area((data[0], data[1]), kdims=[col_name], vdims=['Density']).opts(color=color, alpha=0.5, framewise=True)
            )
        plots.append(hv.Overlay(fplots).opts(shared_axes=True, width=350, height=200))

    for labels, kdim in zip(
        [user_data.selected_emb.labels_cu7, user_data.selected_emb.labels_sos],
        ['Catalog label', 'SOS label'],
    ):
        bar_data = [('', 0)]
        if user_data.selected_emb.has_data:
            bar_data = label_histogram(labels)
        plots.append(
            hv.Bars(
                bar_data, kdims=kdim, vdims=['Count']
            ).opts(width=350, height=250, color='red', framewise=True, xrotation=15)
        )
    layout = hv.Layout(plots).cols(2).opts(shared_axes=False)
    return layout
