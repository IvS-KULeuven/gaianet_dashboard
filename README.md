# GaiaNet dashboard

## Installing and running the dashboard

Clone or download the contents of this repository.

```
git clone https://github.com/IvS-KULeuven/gaianet_dashboard.git
```

Create a virtual environment, install the requirements and download the data. This operation takes about 15m and uses 27GB of disk space but it is done only once.

```
cd gaianet_dashboard
python -m venv dashboard_env
source dashboard_env/bin/activate
pip install -r requirements.txt
dvc pull
```

If you prefer, you can also use a conda environment 

```
cd gaianet_dashboard
conda create -n dashboard_env python=3.12 pip
conda activate dashboard_env
pip install -r requirements.txt
dvc pull
```


(After activating the environment) Launch the dashboard in a browser tab

```
panel serve src/launch_panel.py --show --args data/DR3_40obs_20mag_with_spectra/ data/latent_space/
```

## Using the dashboard

- The scatter on the left shows the embedding. Each dot is as source. You can use the tools to the right of this plot to move and zoom in/out. Colored dots correspond to sources from the CU7 training set.
- Upon using the box selection tool in the embedding plot, the light curves, spectra and sky positions of 12 sources (randomly selected) from the selected region will be shown in the right part of the dashboard. Navigate through the data products using the tabs on the top. The source ids will also appear in the text box in the bottom left from where they can be easily copied.
- The light curve tab can switch between raw light curves and folded light curves by ticking the "Fold" checkbox at the top of the tabs.
- Pressing the green button at the top of the tabs will show a different set of 12 sources from the selected region.

