import time
from io import StringIO
import logging
from functools import partial
import numpy as np
import holoviews as hv
import geoviews as gv
import panel as pn
from holoviews import streams
# from holoviews.selection import link_selections

from metadata import MetadataHandler
from data_loader import DataLoaderSQLite
from user import UserSelection
from plots import lc_layout, dmdt_layout, xp_layout, plot_features, plot_sky, datashade_embedding, datashade_skymap


logger = logging.getLogger(__name__)

colors = [
    'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink',
    'olive', 'cyan', 'magenta', 'yellow', 'teal', 'gold', 'indigo', 'lime',
    'navy', 'coral', 'salmon', 'darkgreen', 'orchid', 'sienna',
    'turquoise', 'maroon', 'darkblue'
]


def build_dashboard(plotter: DataLoaderSQLite,
                    embedding: MetadataHandler,
                    class_metadata: dict,
                    n_rows: int = 3,
                    n_cols: int = 3,
                    ):
    n_plots = n_rows*n_cols
    button_width = 100

    # Pipes for passing source ids
    # ls = link_selections.instance(unselected_alpha=1.0)

    # Embedding plot
    def plot_embedding(x_dim: str,
                       y_dim: str,
                       class_column: str = 'class',
                       class_name: str | None = None,
                       sids: np.ndarray | None = None,
                       **plot_kwargs):
        kdims = [x_dim, y_dim]
        if class_name is None and sids is None:
            return hv.Points(embedding.metadata.select([x_dim, y_dim]).to_numpy(), kdims)
        return hv.Points(
            embedding.filter_embedding(
                sids=sids, class_name=class_name, class_column=class_column
            ), kdims).opts(**plot_kwargs)

    def plot_labeled_embedding(x_dim, y_dim,
                               class_name=None,
                               sids=None):
        kdims = [x_dim, y_dim]
        points = hv.Points(
            embedding.labeled_metadata.to_pandas(),
            kdims, vdims=['macro_class']
        )
        return points.opts(width=650, height=500,
                           color='macro_class', cmap=colors, alpha=0.25,
                           tools=['hover'],
                           legend_position='right', fontsize={'legend': 6})

    emb_columns = embedding.emb_columns
    emb_select = partial(pn.widgets.Select, width=150, options=emb_columns)
    x_sel = emb_select(value=emb_columns[0])
    y_sel = emb_select(value=emb_columns[1])
    emb_stream = {'x_dim': x_sel.param.value, 'y_dim': y_sel.param.value}
    bg_emb = hv.DynamicMap(plot_embedding, streams=emb_stream)

    bg_emb2_dyn = hv.DynamicMap(plot_labeled_embedding, streams=emb_stream)
    #aggregator = ds.by('macro_class', ds.count())
    #bg_emb2_dyn = hd.dynspread(
    #    hd.datashade(bg_emb2, aggregator=aggregator, color_key=colors),
    #    max_px=3, threshold=0.75, shape='circle',
    #).opts(legend_position='right', fontsize={'legend': 6}, width=650, height=500)

    cu7_training_class_sel = pn.widgets.Select(
        name='CU7 training set',
        groups=class_metadata,
        value='none'
    )
    sos_class_sel = pn.widgets.Select(
        name='SOS classification',
        options={'No selection': 'none'} | embedding.sos_classes,
        value='none')

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
    # Copy selection to clipboard
    sids_copy_btn = pn.widgets.Button(
        name="📋 Copy", width=button_width,
        description="Copies the content of the text box to your clipboard"
    )
    sids_copy_btn.js_on_click(
        args={"source": sids_input},
        code="navigator.clipboard.writeText(source.value);"
    )
    # Upload CSV with selection
    sids_upload_btn = pn.widgets.FileInput(
        name="📤 Upload", width=3*button_width, sizing_mode="fixed",
        multiple=False, directory=False, accept='.csv,.txt',
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

    # Clear source id text box button
    sids_clear_btn = pn.widgets.Button(
        name="🗑️ Clear", width=button_width, disabled=False,
        description="Empties the text box"
    )

    def clear_text_box(value):
        sids_input.value = ""
    pn.bind(clear_text_box, value=sids_clear_btn, watch=True)

    # Source selection via embedding plot
    box_selector = streams.BoundsXY(source=bg_emb,
                                    bounds=(-0.1, -0.1, 0.1, 0.1))

    class UserData:
        def __init__(self):
            self.selected_emb = UserSelection()
            self.selected_sos = UserSelection()
            self.selected_cu7 = UserSelection()
            self.last_expr = None
            self.sids = [None]*n_plots
            self.labels = [None]*n_plots
            self.lcs = [None]*n_plots
            self.xps = [None]*n_plots
            self.dmdts = [None]*n_plots
            self.freqs = [None]*n_plots
            self.update_trigger = pn.widgets.Button(visible=False)
            self.resample_trigger = pn.widgets.Button(visible=False)

        def _trigger_update(self):
            self.update_trigger.clicks += 1

        def update_selection_sos(self, class_name: str):
            tinit = time.time()
            self.selected_sos.fill_selection(
                embedding.filter_embedding(
                    class_name=class_name, class_column='SOS_class'
                )
            )
            logger.info(f"Filling SOS selection: {time.time()-tinit:0.4f}")
            self._trigger_update()

        def update_selection_cu7(self, class_name: str):
            tinit = time.time()
            self.selected_cu7.fill_selection(
                embedding.filter_embedding(
                    class_name=class_name, class_column='class'
                )
            )
            logger.info(f"Filling CU7 selection: {time.time()-tinit:0.4f}")
            self._trigger_update()

        def print_select_sids(self):
            if self.selected_emb.has_data:
                sids = self.selected_emb.sids.astype('str')
                sids_txt = '\n'.join(sids.tolist())
                return StringIO(sids_txt)

        def update_selection_via_plot(self, expr):
            # expr = ls.selection_expr
            if expr is None:
                return
            if not str(expr) == str(self.last_expr):
                sids_upload_btn.clear()
                tinit = time.time()
                selection = embedding.filter_embedding_bound(expr)
                n_sources = len(selection)
                # pn.state.notifications.info(f'{n_sources} sources selected.')
                self.selected_emb.fill_selection(selection)
                self.last_expr = expr
                logger.info(f"Update selection via box ({n_sources}): {time.time()-tinit:0.4f}")
                self.resample()
                self._trigger_update()

        def _parse_text(self, text):
            tinit = time.time()
            selection = embedding.validate_uploaded_sources(text)
            if selection is None:
                pn.state.notifications.error('Error parsing the input.')
            else:
                n_sources = len(selection)
                pn.state.notifications.info(f'{n_sources} sources found in the dataset.')
                self.selected_emb.fill_selection(selection)
                logger.info(f"Validate input sids: {time.time()-tinit:0.4f}")
                self.resample()
                self._trigger_update()

        def update_selection_via_upload(self, value):
            if value is None:
                return
            try:
                text = StringIO(value.decode("utf8"))
            except Exception as e:
                logger.error(f"Failed to parse text: {e}")
                pn.state.notifications.error('Error reading the uploaded file.')
                return
            self._parse_text(text)

        def update_selection_via_textbox(self, value):
            sids_textbox = sids_input.value
            try:
                sids_textbox = StringIO(sids_textbox)
            except Exception as e:
                logger.error(f"Failed to parse text: {e}")
                pn.state.notifications.error('Error reading the uploaded file.')
                return
            sids_upload_btn.clear()
            self._parse_text(sids_textbox)

        def resample(self, *args, **kwargs):
            if not self.selected_emb.has_data:
                return
            tinit = time.time()
            sids, freqs, labels = self.selected_emb.random_sample(n_plots)
            self.sids = sids
            sids_input.value = '\n'.join([str(s) for s in sids])
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
            logger.info(f"Update data products: {time.time()-tinit:0.4f}")
            self.resample_trigger.clicks += 1

    user_data = UserData()

    sids_submit_btn = pn.widgets.Button(
        name="✅ Submit", width=button_width,
        description="Selects the source ids in the text box"
    )

    pn.bind(user_data.update_selection_via_textbox, sids_submit_btn, watch=True)
    # Download selection as CSV
    sids_download_btn = pn.widgets.FileDownload(
        callback=user_data.print_select_sids, filename='sids.txt',
        label="⬇️  Download", width=button_width,
        description=(
            "Downloads the content of the text box as a sids.txt file"
        )
    )

    pn.bind(user_data.update_selection_via_upload, sids_upload_btn, watch=True)

    def plot_selection(selection: UserSelection,
                       trigger: bool,
                       x_dim: str,
                       y_dim: str) -> hv.Points:
        kdims = [x_dim, y_dim]
        # Hack to avoid datashader filling the screen with color upon
        # shading and empty array. Can't find a way to do datashading
        # dynamically upon condition
        points = hv.Points([-5, -5], kdims)
        if selection.has_data:
            points = hv.Points(selection.embedding, kdims)
        return points

    sel_emb = hv.DynamicMap(
        pn.bind(partial(plot_selection, selection=user_data.selected_emb),
                trigger=user_data.update_trigger.param.clicks,
                x_dim=x_sel.param.value, y_dim=y_sel.param.value)
    )
    pn.bind(user_data.update_selection_via_plot, box_selector.param.bounds, watch=True)
    sos_class_emb = hv.DynamicMap(
        pn.bind(partial(plot_selection, selection=user_data.selected_sos),
                trigger=user_data.update_trigger.param.clicks,
                x_dim=x_sel.param.value, y_dim=y_sel.param.value)
    )
    pn.bind(user_data.update_selection_sos, sos_class_sel.param.value, watch=True)
    cu7_training_class_emb = hv.DynamicMap(
        pn.bind(partial(plot_selection, selection=user_data.selected_cu7),
                trigger=user_data.update_trigger.param.clicks,
                x_dim=x_sel.param.value, y_dim=y_sel.param.value)
    )
    pn.bind(user_data.update_selection_cu7, cu7_training_class_sel.param.value, watch=True)

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
            f"Chooses {n_plots} sources from the current selection "
            "and plots their light curves and spectra."
        )
    )
    pn.bind(user_data.resample, resample_btn.param.value, watch=True)
    lc_dmap = hv.DynamicMap(
        pn.bind(update_lc_map,
                trigger=user_data.resample_trigger.param.clicks,
                fold=fold_check.param.value,
                highlight_label=cu7_training_class_sel.param.value)
    )
    xp_dmap = hv.DynamicMap(
        pn.bind(update_xp_map,
                trigger=user_data.resample_trigger.param.clicks,
                highlight_label=cu7_training_class_sel.param.value)
    )
    dmdt_dmap = hv.DynamicMap(
        pn.bind(update_dmdt_map,
                trigger=user_data.resample_trigger.param.clicks,
                highlight_label=cu7_training_class_sel.param.value)
    )

    def update_feature_map(user_data, trigger):
        tinit = time.time()
        layout = plot_features(user_data)
        logger.info(f"Building feature plot: {time.time()-tinit:0.4f}")
        return layout
    feature_dmap = hv.DynamicMap(
        pn.bind(partial(update_feature_map, user_data=user_data),
                trigger=user_data.update_trigger.param.clicks)
    )

    # Sky map
    sky = gv.Points(embedding.metadata_hv, kdims=['longitude', 'latitude'])
    bg_sky = datashade_skymap(sky)

    def update_sky_selection(selection, color, trigger):
        tinit = time.time()
        if selection.has_data:
            plot = plot_sky(selection.coordinates, color=color)
        else:
            plot = plot_sky([], color=color)
        logger.info(f"Update sky map: {time.time()-tinit:0.4f}")
        return plot

    sky_cu7 = hv.DynamicMap(
        pn.bind(
            partial(update_sky_selection, selection=user_data.selected_cu7, color='blue'),
            trigger=user_data.update_trigger.param.clicks
        )
    )
    sky_sos = hv.DynamicMap(
        pn.bind(
            partial(update_sky_selection, selection=user_data.selected_sos, color='green'),
            trigger=user_data.update_trigger.param.clicks
        )
    )
    sky_sel = hv.DynamicMap(
        pn.bind(
            partial(update_sky_selection, selection=user_data.selected_emb, color='red'),
            trigger=user_data.update_trigger.param.clicks
        )
    )
    # sky_cu7 = datashade_skymap(sky_cu7, cmap='blues')
    # sky_sel = datashade_skymap(sky_sel, cmap='reds')
    # sky_sos = datashade_skymap(sky_sos, cmap='greens')

    # Dashboard
    tabs = pn.Tabs(
        ('Light curves', lc_dmap),
        ('Sampled spectra', xp_dmap),
        ('DMDT', dmdt_dmap),
        ('Sky map', hv.Overlay([
            bg_sky * sky_cu7 * sky_sos * sky_sel,
        ]).collate()),
        ('Features', feature_dmap),
        dynamic=True
    )
    bg_emb = datashade_embedding(bg_emb, cmap='gray')
    sos_class_emb = datashade_embedding(sos_class_emb, cmap='greens')
    sel_emb = datashade_embedding(sel_emb, cmap='reds')
    cu7_training_class_emb = datashade_embedding(cu7_training_class_emb, cmap='blues')
    # bg_emb = ls(bg_emb)
    emb_overlay = hv.Overlay([bg_emb, sos_class_emb, sel_emb, cu7_training_class_emb])
    return pn.Row(
        pn.Column(
            pn.Tabs(('Unlabeled', emb_overlay.collate()),
                    ('Labeled', bg_emb2_dyn), dynamic=True),
            # pn.Row(x_sel, y_sel, cu7_training_class_sel),
            pn.Row(sos_class_sel, cu7_training_class_sel),
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
