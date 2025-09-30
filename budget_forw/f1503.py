import seaduck.lagrangian_budget as sdlb
import seaduck as sd
import time
import numpy as np
import xarray as xr
from open4dense import *
import os

special = 'f1503to1603'
start_time = '2015-03-01'
end_time = '2016-03-01'
nep_path = '/sciserver/filedb04-01/ocean/wenrui_temp/particle_file/NEP/'
path = nep_path+special+'/'

nt = len(os.listdir(path))

wallt_path = '/sciserver/filedb09-01/ocean/wall_theta.zarr'
data_path = '/sciserver/filedb02-02/ocean/wenrui_temp/heat/'
ds = give_me_ecco_heat_seas(data_path,wallt_path,time_mean_vel = False)
tp = sd.Topology(ds)


ds['count']= xr.DataArray(np.arange(len(ds.time)),dims = 'time')
idate = int(ds['count'].sel(time = start_time))
idate_last = int(ds['count'].sel(time = end_time))
assert idate_last == idate+(nt)
dataset_date_id = np.array(ds.time[idate:idate+nt])
to_isel = np.arange(idate,idate+nt)

rhs_list = ['e_ua','E','dif_h','dif_v','A','I','F']

files = sorted([path+i for i in os.listdir(path)])
for i in [1,5,-1]:
    print('checking compatibility: ',i,'...',end = '')
    xrslc = ds.isel(time = to_isel[i])
    example = xr.open_zarr(files[i])
    ans,_ = sdlb.check_particle_data_compat(example,xrslc,tp,wall_names=('txprime','typrime','tzprime'),conv_name = 'divuty')
    assert ans
    print('pass')
Np = len(example.shapes)

def cumu_map_array(neo, value):
    ind = tuple(np.array(i)[:-1] for i in [neo.iz-1, neo.face, neo.iy, neo.ix])
    array = np.zeros((50,13,90,90))
    np.add.at(array, ind, value)
    return array

maps = {}
for var in rhs_list+['count']:
    maps[var] = np.zeros((50,13,90,90))

hovmoller = xr.Dataset()
for var in rhs_list+['lhs', 'sf', 'sl', 'lon', 'lat', 'dep']:
    hovmoller[var] = xr.DataArray(np.zeros((len(dataset_date_id),Np)),dims = ('time','space'))

t1 = time.time()
for it,(dataset_date, particle_name) in enumerate(zip(to_isel,files)):
    my = ds.isel(time = dataset_date)
    if it %10==0:
        print('dataset_date:', particle_name, 
              str(it)+'/'+str(nt), 
              time.time()-t1)
    neo = xr.open_zarr(particle_name)
    prefetch_vector_kwarg = dict(xname="txprime", yname="typrime", zname="tzprime")
    contr_dic, s_wall, first, last = sdlb.calculate_budget(neo, my,rhs_list, prefetch_vector_kwarg, dir_of_time=1)

    for var in rhs_list+['lhs']:
        cumsum = np.cumsum(np.nan_to_num(contr_dic[var]))
        cumsum = np.insert(cumsum[last-1],0,0)
        hovmoller[var][it,:] = np.diff(cumsum)
        # assert np.isclose(np.nansum(hovmoller[var][it]),np.nansum(contr_dic[var]))
    hovmoller['sl'][it,:] = s_wall[last]
    hovmoller['sf'][it,:] = s_wall[first]
    hovmoller['lon'][it,:] = np.array(neo.xx[first])
    hovmoller['lat'][it,:] = np.array(neo.yy[first])
    hovmoller['dep'][it,:] = np.array(neo.zz[first])
    for var in rhs_list:
        maps[var] += cumu_map_array(neo, contr_dic[var])
    maps['count'] += cumu_map_array(neo, np.ones_like(contr_dic['A']))

maps_ds = xr.Dataset()
for var in rhs_list+['count']:
    maps_ds[var] = xr.DataArray(maps[var],dims = ('Z','face','Y','X'))

map_path = nep_path+ 'map_'+special+'.zarr'
table_path = nep_path+'table_'+special+'.zarr'
maps_ds.to_zarr(map_path, mode = 'w')
hovmoller.to_zarr(table_path, mode = 'w')
print('success')