# GaiaNet dashboard

Clone or download the contents of this repository.

```
git clone https://github.com/IvS-KULeuven/gaianet_dashboard.git
```

Create a virtual environment, install the requirements and download the data. This about 15m but it is done only once.

```
cd gaianet_dashboard
python -m venv dashboard_env
source dashboard_env/bin/activate
pip install -r requirements.txt
dvc pull
```

Launch the dashboard in a browser tab

```
panel serve src/launch_panel.py --show --args data/DR3_40obs_20mag_with_spectra/ data/latent_space/2
```


