# -*- coding: utf-8 -*-
import climsr.consts as consts

cruts_to_world_clim_mapping = {
    consts.cruts.tmn: consts.world_clim.tmin,
    consts.cruts.tmp: consts.world_clim.temp,
    consts.cruts.tmx: consts.world_clim.tmax,
    consts.cruts.pre: consts.world_clim.prec,
}

world_clim_to_cruts_mapping = dict(
    [(v, k) for k, v in cruts_to_world_clim_mapping.items()]
)
