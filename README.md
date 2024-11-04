# GaiaNet dashboard

## Installing and downloading data

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

If you prefer, you can also use a conda/mamba environment 

```
cd gaianet_dashboard
conda create -n dashboard_env python=3.11 pip
conda activate dashboard_env
pip install -r requirements.txt
dvc pull
```

## Launching and using the dashboard

(After activating the environment) Launch the dashboard in a browser tab

```
python -m panel serve src/launch_panel.py --show --args data data/latent_space/vae.csv
```


- The scatter on the left shows the embedding. Each dot is a source. You can use the tools to the right of this plot to move and zoom in/out.
- Upon using the box selection tool in the embedding plot, the light curves and spectra of 12 sources (randomly selected) from the selected region will be shown in the right part of the dashboard. Navigate through the data products using the tabs on the top. The source ids will also appear in the text box in the bottom left, from where they can be easily copied or downloaded.
- The light curve tab can switch between raw light curves and folded light curves by ticking the "Fold" checkbox at the top of the tabs.
- Pressing the "Resample" button at the bottom left of the dashboard will show a different set of 12 sources from the selected region.

The light curve tab:

![Screenshot 2024-10-02 at 10-21-04 GaiaNet embedding explorer](https://github.com/user-attachments/assets/d140e4f3-63a4-4350-a1cf-c8a5f3003252)

The light curve tab with the folded option activated:

![Screenshot 2024-10-02 at 10-21-14 GaiaNet embedding explorer](https://github.com/user-attachments/assets/37a465fc-3e5e-4bd0-9616-43f046c49dd9)

The sampled xp spectra tab:

![Screenshot 2024-10-02 at 10-21-22 GaiaNet embedding explorer](https://github.com/user-attachments/assets/461c2eec-3d95-4e34-9c70-853317977f03)

The sky tab:

![Screenshot 2024-10-02 at 10-21-31 GaiaNet embedding explorer](https://github.com/user-attachments/assets/d579c177-9fd4-46f1-839b-9e7fae8b825e)

## Exploring the data without the dashboard

[This notebook](notebooks/demo.ipynb) shows how to obtain metadata, embedding positions and data products for a given source id.

