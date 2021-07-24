# -*- coding: utf-8 -*-
from sr.configs.cruts_config import CRUTSConfig
from sr.configs.world_clim_config import WorldClimConfig

cruts_to_world_clim_mapping = {
    CRUTSConfig.tmn: WorldClimConfig.tmin,
    CRUTSConfig.tmp: WorldClimConfig.temp,
    CRUTSConfig.tmx: WorldClimConfig.tmax,
    CRUTSConfig.pre: WorldClimConfig.prec,
}

world_clim_to_cruts_mapping = dict(
    [(v, k) for k, v in cruts_to_world_clim_mapping.items()]
)
