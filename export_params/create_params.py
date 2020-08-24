import json
params = { 
    "collections" : {
        "multispectral" : "ASK",
        "dsm" : "ASK",
        "tropomi" : "COPERNICUS/S5P/NRTI/L3_NO2",
        "road" : "ASK",
        "wind" : "ASK"},
    "bands": {
        "ortho": ["B", "G", "R", "N", "P"],
        "tropomi_NO2": ["tropospheric_NO2_column_number_density", "cloud_fraction"],
        "road":["num_observations"],
        "dsm": ["dsm"],
        "wind":["u_component_of_wind_10m", "v_component_of_wind_10m"]},
    "kernel_radius": 128,
    "scale": 50
}

json.dump(params, open("parameters.json", 'w'))
