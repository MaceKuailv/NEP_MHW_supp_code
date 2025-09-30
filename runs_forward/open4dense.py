import numpy as np
import xarray as xr
from itertools import accumulate
import os

for i in range(1,13):
    for j in range(1,4):
        new_path = f'/sciserver/filedb{i:02}-0{j}/'
        os.listdir(new_path)

def give_me_ecco(path0,path1,path2,path,walls_path,time_mean_vel = True):
    advdif = xr.open_zarr(path2+'advNdif_normal')
    tends = xr.open_zarr(path1+'tendency_normal')
    forcs = xr.open_zarr(path2+'saltfluxandplume')
    rain = xr.open_zarr(path0+'RainForcing')
    err_m = xr.open_zarr(path2+'Error_mean')
    advdif_m = xr.open_zarr(path0+'advNdif_mean')
    rain_m = xr.open_zarr(path1+'RainForcing_mean')
    divus = xr.open_zarr(path+'divus.zarr')
    euas = xr.open_zarr(path+'e_ua_s.zarr')
    # sdivu = xr.open_zarr(path+'sdivu.zarr')
    e_ssh = xr.open_zarr(path+'e_ssh_s.zarr')
    upgrdsp_m = xr.open_zarr(path+'upgrdsp_m.zarr')
    smdivum = xr.open_zarr(path+'smdivum.zarr')
    euas_m = xr.open_zarr(path+'e_ua_s_m.zarr')
    e_ssh_m = xr.open_zarr(path+'e_ssh_s_m.zarr')
    spdivum = xr.open_zarr(path+'spdivum.zarr')
    smdivup = xr.open_zarr(path+'smdivup.zarr')
    spdivup = xr.open_zarr(path+'spdivup.zarr')
    walls_m = xr.open_zarr(path+'walls_m.zarr')
    walls = xr.open_zarr(walls_path)
    
    ds = xr.merge([
        advdif,tends,forcs,rain,
        err_m,advdif_m,rain_m,
        divus,euas,e_ssh,spdivum,smdivup,spdivup,
        upgrdsp_m,smdivum,euas_m,
        walls,walls_m,
    ])
    
    # Slightly modify the tendency term for consistency
    tend_ = xr.open_zarr(path1+'tendS_0N-1')
    ds.tendS[0] = tend_.tendS_first
    ds.tendS[-1]= tend_.tendS_last
    
    ds['e_ua'] = (euas.e_ua_s-euas_m.e_ua_s_m)
    ds['e_ssh'] = (e_ssh.e_ssh_s-e_ssh_m.e_ssh_s_m)
    if time_mean_vel:
        ds['E'] = (divus.divupsp-spdivup.spdivup-upgrdsp_m.upgrdsp_m)
        ds['R'] = -(divus.divumsp-spdivum.spdivum)
    else:
        ds['E'] = (-upgrdsp_m.upgrdsp_m)
        ds['R'] = -(divus.divumsp-spdivum.spdivum+divus.divupsp-spdivup.spdivup)
    ds['dif_h'] = (ds.dif_hConvS-ds.dif_hConvS_mean).transpose('time','Z','face','Y','X')
    ds['dif_v'] = (ds.dif_vConvS-ds.dif_vConvS_mean).transpose('time','Z','face','Y','X')
    ds['I'] = (ds.forcS-ds.forcS_mean)
    ds['A'] = (divus.divupsm-smdivup.smdivup)
    ds['U'] = ds.tendS_mean - ds.tendS
    ds['F'] = (ds.pe_mean - ds.pe)
    ds['lhs'] = ds['U'] + ds['e_ssh']
    
    ds['sxprime'] = ds['sx'] - ds['sxm']
    ds['syprime'] = ds['sy'] - ds['sym']
    ds['szprime'] = ds['sz'] - ds['szm']
    
    rhs_list = ['e_ua','E','dif_h','dif_v','A','I','F']
    termlist = rhs_list+['lhs']
    for var in termlist:
        if var == 'E' and not time_mean_vel:
            continue
        ds[var] = ds[var].transpose('time','Z','face','Y','X')
    return ds

def give_me_ecco_heat(path,wallt_path,time_mean_vel = True):
    adv = xr.open_zarr(path+'adv.zarr')
    dif_h = xr.open_zarr(path+'dif_h.zarr')
    dif_v = xr.open_zarr(path+'dif_v.zarr')
    forch = xr.open_zarr(path+'forcH.zarr')
    tendh = xr.open_zarr(path+'tendH.zarr')

    divut = xr.open_zarr(path+'divut.zarr')
    euat = xr.open_zarr(path+'e_ua_t.zarr')
    tdivu = xr.open_zarr(path+'tdivu.zarr')
    e_ssh = xr.open_zarr(path+'e_ssh_t.zarr')
    upgrdtp_m = xr.open_zarr(path+'upgrdtp_m.zarr')
    tmdivum = xr.open_zarr(path+'tmdivum.zarr')
    euat_m = xr.open_zarr(path+'e_ua_t_m.zarr')
    e_ssh_m = xr.open_zarr(path+'e_ssh_t_m.zarr')
    tpdivum = xr.open_zarr(path+'tpdivum.zarr')
    tmdivup = xr.open_zarr(path+'tmdivup.zarr')
    tpdivup = xr.open_zarr(path+'tpdivup.zarr')
    tforcv = xr.open_zarr(path+'tforcv.zarr')
    
    tforcv_m = xr.open_zarr(path+'tforcv_m.zarr')
    adv_m = xr.open_zarr(path+'adv_m.zarr')
    dif_h_m = xr.open_zarr(path+'dif_h_m.zarr')
    dif_v_m = xr.open_zarr(path+'dif_v_m.zarr')
    tendhm = xr.open_zarr(path+'tendH_m.zarr')
    forchm = xr.open_zarr(path+'forcH_m.zarr')

    tend_ = xr.open_zarr(path+'tend1N-1.zarr')

    wallt_m = xr.open_zarr(path+'wallt_m.zarr')
    wallt = xr.open_zarr(wallt_path)

    ds = xr.merge([divut,tendh,wallt,wallt_m])

    ds.tendH[0] = tend_.tendH_first
    ds.tendH[-1]= tend_.tendH_last
    
    ds['U'] = -(tendh['tendH']-tendhm['tendH_m'])
    ds['A'] = (divut.divuptm-tmdivup.tmdivup)
    if time_mean_vel:
        ds['E'] = (divut.divuptp-tpdivup.tpdivup-upgrdtp_m.upgrdtp_m)
        ds['R'] = -(divut.divumtp-tpdivum.tpdivum)
    else:
        ds['E'] = (-upgrdtp_m.upgrdtp_m)
        ds['R'] = -(divut.divumtp-tpdivum.tpdivum+divut.divuptp-tpdivup.tpdivup)
    ds['F'] = -(tforcv_m.tforcv_m - tforcv.tforcv)
    ds['e_ssh'] = (e_ssh.e_ssh_t-e_ssh_m.e_ssh_t_m)
    ds['e_ua'] = (euat.e_ua_t-euat_m.e_ua_t_m)
    ds['dif_h'] = (dif_h.dif_h-dif_h_m.dif_h_m)
    ds['dif_v'] = (dif_v.dif_v-dif_v_m.dif_v_m)
    ds['I'] = (forch.forcH-forchm.forcH_m)
    
    ds['lhs'] = ds['U'] + ds['e_ssh']
    
    ds['txprime'] = (ds['tx'] - ds['txm']).transpose('time','Z','face','Y','Xp1')
    ds['typrime'] = (ds['ty'] - ds['tym']).transpose('time','Z','face','Yp1','X')
    ds['tzprime'] = (ds['tz'] - ds['tzm']).transpose('time','Zl','face','Y','X')

    rhs_list = ['e_ua','E','dif_h','dif_v','A','I','F']
    termlist = rhs_list+['lhs']
    for var in termlist:
        if var == 'E' and not time_mean_vel:
            continue
        ds[var] = ds[var].transpose('time','Z','face','Y','X')
    return ds
    

def give_me_orig_ecco():
    mean = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/mean*',engine = 'zarr')
    snap = xr.open_mfdataset('/sciserver/filedb0*-02/ocean/poseidon/daily_mean_ecco/zarr/snap*',engine = 'zarr')
    grid = xr.open_zarr('~/ECCO_transport')
    bc = xr.open_zarr('/sciserver/filedb09-01/ocean/GM_vel.zarr') #bolus correct
    ds = xr.merge([mean,bc, snap])
    ds = ds.reset_coords().assign_coords(grid.coords).astype(float)
    return ds

def merge_particle_files(files,date,region_names=False):
    assert np.array([(date in name) for name in files]).all(), (date, files[0])
    datasets = [xr.open_zarr(files[i]) for i in range(len(files))]
    ds0 = datasets[0]
    neo = xr.Dataset()
    neo['shapes'] = xr.concat([ds.shapes for ds in datasets], dim = 'shapes')
    nprof = [len(ds.nprof) for ds in datasets]
    prefix = [0]+list(accumulate(nprof))
        
    for var in ['face','frac','ind1','ind2','ix','iy','iz','tres','tt','vs','xx','yy','zz']:
        neo[var] = xr.concat([ds[var] for ds in datasets], dim = 'nprof')
    if region_names:
        for region in region_names:
            neo[region] = xr.concat([ds[region] for ds in datasets], dim = 'nprof')
    return neo

def give_me_ecco_heat_seas(path,wallt_path,time_mean_vel = True):
    adv = xr.open_zarr(path+'adv.zarr')
    dif_h = xr.open_zarr(path+'dif_h.zarr')
    dif_v = xr.open_zarr(path+'dif_v.zarr')
    forch = xr.open_zarr(path+'forcH.zarr')
    tendh = xr.open_zarr(path+'tendH.zarr')

    divut = xr.open_zarr(path+'divut.zarr')
    euat = xr.open_zarr(path+'e_ua_t.zarr')
    tdivu = xr.open_zarr(path+'tdivu.zarr')
    e_ssh = xr.open_zarr(path+'e_ssh_t.zarr')
    
    tenj = xr.open_zarr(path+'tendH_j.zarr')
    advj = xr.open_zarr(path+'adv_j.zarr')
    difhj = xr.open_zarr(path+'dif_h_j.zarr')
    difvj = xr.open_zarr(path+'dif_v_j.zarr')
    forcj = xr.open_zarr(path+'forcH_j.zarr')
    
    uygrdty_j = xr.open_zarr(path+'uygrdty_j.zarr')
    tyuy = xr.open_zarr(path+'tydivuy.zarr')
    tjuj = xr.open_zarr(path+'tjdivuj.zarr')
    tyuj = xr.open_zarr(path+'tydivuj.zarr')
    tjuy = xr.open_zarr(path+'tjdivuy.zarr')

    esshj = xr.open_zarr(path+'e_ssh_t_j.zarr')
    tforcj = xr.open_zarr(path+'tforcv_j.zarr')
    euaj = xr.open_zarr(path+'e_ua_t_j.zarr')
    tforcv = xr.open_zarr(path+'tforcv.zarr')

    seas = xr.open_zarr(path+'divut_seas.zarr')

    tend_ = xr.open_zarr(path+'tend1N-1.zarr')

    txj = xr.open_zarr(path+'txj.zarr')
    tyj = xr.open_zarr(path+'tyj.zarr')
    tzj = xr.open_zarr(path+'tzj.zarr')
    wallt_j = xr.merge([txj,tyj,tzj])
    wallt = xr.open_zarr(wallt_path)

    ds = xr.merge([divut,tendh,wallt,wallt_j])

    ds.tendH[0] = tend_.tendH_first
    ds.tendH[-1]= tend_.tendH_last
    
    ds['U'] = -(tendh['tendH'].groupby('time.dayofyear')-tenj['tendH_j'])
    ds['A'] = (seas.divuytj-tjuy.tjdivuy)
    if time_mean_vel:
        raise NotImplemented('I am a bit lazy')
        # ds['E'] = (divut.divuptp-tpdivup.tpdivup-upgrdtp_m.upgrdtp_m)
        # ds['R'] = -(divut.divumtp-tpdivum.tpdivum)
    else:
        ds['E'] = (-uygrdty_j.uygrdty_j)*xr.ones_like(ds['U']).groupby('time.dayofyear')
        ds['R'] = -(seas.divujty-tyuj.tydivuj+seas.divuyty-tyuy.tydivuy)
    ds['F'] = (tforcj.tforcv_j - tforcv.tforcv.groupby('time.dayofyear'))
    ds['e_ssh'] = (e_ssh.e_ssh_t.groupby('time.dayofyear')-esshj.e_ssh_t_j)
    ds['e_ua'] = (euat.e_ua_t.groupby('time.dayofyear')-euaj.e_ua_t_j)
    ds['dif_h'] = (dif_h.dif_h.groupby('time.dayofyear')-difhj.dif_h_j)
    ds['dif_v'] = (dif_v.dif_v.groupby('time.dayofyear')-difvj.dif_v_j)
    ds['I'] = (forch.forcH.groupby('time.dayofyear')-forcj.forcH_j)
    
    ds['lhs'] = ds['U'] + ds['e_ssh']
    
    ds['txprime'] = (ds['tx'].groupby('time.dayofyear') - ds['txj']).transpose('time','Z','face','Y','Xp1')
    ds['typrime'] = (ds['ty'].groupby('time.dayofyear') - ds['tyj']).transpose('time','Z','face','Yp1','X')
    ds['tzprime'] = (ds['tz'].groupby('time.dayofyear') - ds['tzj']).transpose('time','Zl','face','Y','X')

    rhs_list = ['e_ua','E','dif_h','dif_v','A','I','F']
    termlist = rhs_list+['lhs','R']
    for var in termlist:
        if 'dayofyear' in ds[var].dims:
            print(var)
            raise
        if var == 'E' and not time_mean_vel:
            continue
        ds[var] = ds[var].transpose('time','Z','face','Y','X')
    return ds