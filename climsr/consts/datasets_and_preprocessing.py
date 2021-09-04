# -*- coding: utf-8 -*-
filename = "filename"
file_path = "file_path"
year = "year"
dataset = "dataset"
variable = "variable"
x = "x"
y = "y"
month = "month"
train_csv = "train.csv"
val_csv = "val.csv"
test_csv = "test.csv"
tile_file_path = "tile_file_path"
stage = "stage"
multiplier = "multiplier"

europe_bbox_lr = ((-16.0, 84.5), (40.5, 33.0))
europe_bbox_hr = ((-16.0, 84.5), (40.5, 33.0))
left_upper_lr = [-16.0, 84.5]
left_lower_lr = [-16.0, 33.0]
right_upper_lr = [40.5, 84.5]
right_lower_lr = [40.5, 33.0]

left_upper_hr = [-16.0, 84.5]
left_lower_hr = [-16.0, 33.0]
right_upper_hr = [40.5, 84.5]
right_lower_hr = [40.5, 33.0]

lr_polygon = [
    [
        left_upper_lr,
        right_upper_lr,
        right_lower_lr,
        left_lower_lr,
        left_upper_lr,
    ]
]
hr_polygon = [
    [
        left_upper_hr,
        right_upper_hr,
        right_lower_hr,
        left_lower_hr,
        left_upper_hr,
    ]
]

var_to_variable = {
    "pre": "Precipitation",
    "tmn": "Minimum Temperature",
    "tmp": "Average Temperature",
    "tmx": "Maximum Temperature",
}

lr_bbox = [
    {
        "coordinates": lr_polygon,
        "type": "Polygon",
    }
]

hr_bbox = [
    {
        "coordinates": hr_polygon,
        "type": "Polygon",
    }
]

cruts_to_world_clim_mapping = {
    "tmn": "tmin",
    "tmp": "temp",
    "tmx": "tmax",
    "pre": "prec",
}

world_clim_to_cruts_mapping = dict([(v, k) for k, v in cruts_to_world_clim_mapping.items()])
cruts_download_dir = "cruts"
world_clim_download_dir = "world-clim"
archives = "archives"
extracted = "extracted"
world_clim_main_extraction_folder = "wc2.1"
