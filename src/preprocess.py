import numpy as np
import polars as pl


def col_to_array(col: pl.Series) -> np.ndarray:
    return col[0].to_numpy().astype('float32')


def pack_light_curve(df_row: pl.DataFrame,
                     remove_extreme_errors: bool
                     ) -> dict[str, np.ndarray]:
    light_curve = {}
    bands = [col.split('_')[0] for col in df_row.columns if 'obstimes' in col]
    for band in bands:
        time = col_to_array(df_row[f'{band}_obstimes'])
        val = col_to_array(df_row[f'{band}_val'])
        valerr = col_to_array(df_row[f'{band}_valerr'])
        lcb = np.stack([time, val, valerr])
        mask = ~np.isinf(valerr) & ~np.isnan(valerr)
        lcb = lcb[:, mask]
        if remove_extreme_errors:
            valerr = valerr[mask]
            mask = ~(valerr > valerr.mean() + 5*valerr.std())
            lcb = lcb[:, mask]
        idx = lcb[0].argsort()
        light_curve[band] = lcb[:, idx]
    return light_curve


def pack_spectra(df_row: pl.DataFrame,
                 ) -> dict[str, np.ndarray]:
    xp_spectra = {}
    for band in ['bp', 'rp']:
        values = []
        values.append(
            col_to_array(df_row[f'{band}_val'])
        )
        keys = df_row.columns
        if f'{band}_val_std' in keys:
            values.append(
                col_to_array(df_row[f'{band}_val_std'])
            )
        if f'{band}_val_skew' in keys:
            values.append(
                col_to_array(df_row[f'{band}_val_skew'])
            )
        if f'{band}_valerr' in keys:
            values.append(
                col_to_array(df_row[f'{band}_valerr'])
            )
        xp_spectra[band] = np.stack(values)
    return xp_spectra
