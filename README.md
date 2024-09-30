# GaiaNet dashboard

Clone or download the contents of this repository.

```
git clone https://github.com/IvS-KULeuven/gaianet_dashboard.git
```

Create a virtual environment, install the requirements and download the data. This takes about 15m but it is done only once.

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
panel serve src/launch_panel.py --show --args data/DR3_40obs_20mag_with_spectra/ data/latent_space/2
```
