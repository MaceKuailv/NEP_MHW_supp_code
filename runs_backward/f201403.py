import numpy as np
from open4dense import give_me_orig_ecco
import seaduck as sd
import xarray as xr
from seaduck.get_masks import which_not_stuck
from seaduck.eulerian_budget import _left90, _right90
from scipy.ndimage import label
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

save_path = filedb_lst[9]+'/ocean/wenrui_temp/particle_file/NEP/f1403to1309/'
path = '/sciserver/filedb02-02/ocean/wenrui_temp/heat/'

ds = give_me_orig_ecco()
ds['utrans'] = (ds['u_gm']+ds['UVELMASS'])*ds.dyG*ds.drF
ds['vtrans'] = (ds['v_gm']+ds['VVELMASS'])*ds.dxG*ds.drF
ds['wtrans'] = (ds['w_gm']+ds['WVELMASS'])*ds.rA
tseas1 = xr.open_zarr(path+'tseas1.zarr')
tseas2 = xr.open_zarr(path+'tseas2.zarr')
tseas3 = xr.open_zarr(path+'tseas3.zarr')
tseas = xr.concat([tseas1,tseas2,tseas3],dim = 'dayofyear')
ta = (ds.THETA.groupby('time.dayofyear') - tseas).transpose('time','Z','face','Y','X').THETA

tub = sd.OceData(ds)

time = '2014-03-01'
t = sd.utils.convert_time(time)
end_time = '2013-09-01'
end_time = sd.utils.convert_time(end_time)
stops = np.array([end_time])

ylim = (32.5,55)
xlim = (160,255)

xbool = np.logical_and(ds.XC%360>xlim[0],ds.XC%360<xlim[1])
ybool = np.logical_and(ds.YC>ylim[0],ds.YC<ylim[1])
pos_bool = np.logical_and(np.logical_and(xbool,ybool),ds.Z>-200).transpose('Z','face','Y','X')
warm_bool = ta.sel(time = time)[0]>1.25
those = np.logical_and(warm_bool,pos_bool)
merged = merge_ep(those)
labeled_array, num_features = label(merged)
volumes = np.bincount(labeled_array.ravel())
label2select = np.argmax(volumes[1:])+1
largest_blob = labeled_array==label2select

these = split_merge(largest_blob)
ds['bool'] = xr.DataArray(these,dims = ds.HFacC.dims)
these = ds['bool']*(ds.drF*ds.rA*ds.HFacC)
vol = float(np.sum(these))
num = int(vol/3e10)
print(num)

p = sd.Particle(
    bool_array=ds['bool'], num=num, random_seed=seed,
    t = t,
    data = tub, free_surface = 'noflux',
    save_raw = True,uname = 'utrans',vname = 'vtrans',wname = 'wtrans',
    transport  = True
)
p=p.subset(sd.get_masks.which_not_stuck(p))
p.empty_lists()

print('finished pre-calculating',p.N)

p.to_list_of_time(normal_stops = stops,dump_filename = save_path+f'Seed{seed}_',store_kwarg = {'preserve_checks':True})
print('success', p.N)