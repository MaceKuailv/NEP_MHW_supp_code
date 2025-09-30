import numpy as np
from open4dense import give_me_orig_ecco
import seaduck as sd
import xarray as xr
from seaduck.get_masks import which_not_stuck
from seaduck.eulerian_budget import _left90, _right90
from scipy.ndimage import label
import e4p
import os 

def merge_ep(da):
    merged = np.zeros((50,180,180))
    merged[:,:90,:90] = _right90(np.array(da[:,8] ))
    merged[:,90:,:90] = _right90(np.array(da[:,7] ))
    merged[:,:90,90:] = _right90(np.array(da[:,11]))
    merged[:,90:,90:] = _right90(np.array(da[:,10]))
    return merged

def split_merge(merged):
    da = np.zeros((50,13,90,90))
    da[:,8] = _left90(merged[:,:90,:90])
    da[:,7] = _left90(merged[:,90:,:90])
    da[:,11]= _left90(merged[:,:90,90:])
    da[:,10]= _left90(merged[:,90:,90:])
    return da

filedb_lst = []
for i in range(1,13):
    for j in range(1,4):
        filedb_lst.append(f'/sciserver/filedb{i:02}-0{j}')

seed = 2014

save_path = filedb_lst[9]+'/ocean/wenrui_temp/particle_file/NEP/south_mhw/'
path = '/sciserver/filedb02-02/ocean/wenrui_temp/heat/'

ds = give_me_orig_ecco()
ds['utrans'] = (ds['u_gm']+ds['UVELMASS'])*ds.dyG*ds.drF
ds['vtrans'] = (ds['v_gm']+ds['VVELMASS'])*ds.dxG*ds.drF
ds['wtrans'] = (ds['w_gm']+ds['WVELMASS'])*ds.rA

tub = sd.OceData(ds)

time = '2016-03-01'
t = sd.utils.convert_time(time)
end_time = '2013-09-01'
end_time = sd.utils.convert_time(end_time)
stops = np.array([end_time])

conc = np.load('../all_mhw.npy')
conc = conc*(conc>0.1)
# conc = e4p.llc_compact_to_tiles(conc)

num = int(1e5)

p = sd.Particle(
    bool_array=conc, num=num, random_seed=seed,
    t = t-0.1,
    data = tub, free_surface = 'noflux',
    save_raw = True,uname = 'utrans',vname = 'vtrans',wname = 'wtrans',
    transport  = True
)
p=p.subset(sd.get_masks.which_not_stuck(p))
p = p.subset(p.lat<=30)
p.empty_lists()

print('finished pre-calculating',p.N)

p.to_list_of_time(normal_stops = stops,dump_filename = save_path+f'Seed{seed}_',store_kwarg = {'preserve_checks':True})
print('success', p.N)