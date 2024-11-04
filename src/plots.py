from typing import Callable
from functools import partial
import numpy as np
import holoviews as hv
import holoviews.operation.datashader as hd


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
                    height: int = 160,
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
                 height: int = 160,
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


def plot_dmdt(data: np.ndarray | None,
              sid: str | None,
              label: str | None,
              width: int = 200,
              height: int = 220,
              **kwargs,
              ):
    title = format_title(sid, label)
    if data is None:
        data = []
    return hv.Image(data).opts(
        title=title, fontsize={'title': 6}, tools=['hover'],
        cmap='blues', clim=(0, 1), framewise=True, width=width, height=height
    )


lc_layout = partial(make_layout, plot_function=plot_lightcurve)
xp_layout = partial(make_layout, plot_function=plot_spectra)
dmdt_layout = partial(make_layout, plot_function=plot_dmdt)


# TODO: MAKE DYNAMIC MAP SCALE THE AXES
def datashade_embedding(emb_plot, cnorm: str = "eq_hist"):
    return hd.dynspread(
        hd.datashade(emb_plot, cmap="gray", cnorm=cnorm),
        max_px=3, threshold=0.75, shape='circle',
    ).opts(width=650, height=500, tools=['box_select'],
           active_tools=['box_select', 'wheel_zoom'])
