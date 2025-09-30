import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import numpy as np

dpi = 300
rerun = True
regen_talk = False

nep_extent = [110, 250, 20, 65]
projection = ccrs.Mercator(central_longitude=180.0, 
                           min_latitude=20.0, 
                           max_latitude=80.0,
                           latitude_true_scale=40.0)

balance = cmocean.cm.balance
tempo = cmocean.cm.tempo
depth_cmap = "Greys_r"
depth_norm = mpl.colors.Normalize(vmin=-5000, vmax=5000)

nep_time_cmap = plt.get_cmap('BuPu_r')
nep_theme_color = 'teal'
nep_idate = 8643

wau_time_cmap = plt.get_cmap('OrRd_r')
wau_theme_color = 'maroon'
wau_idate = 6999

mean_time_cmap = plt.get_cmap('YlGn_r')

s_cmap = plt.get_cmap('PuOr_r')
term_cmap = balance
term_cmap_r = cmocean.cm.balance_r

# a_palette5 = ["#df2935","#86ba90","#f5f3bb","#dfa06e","#412722"]
# a_palette5 = ["#1b4079","#4d7c8a","#7f9c96","#8fad88","#cbdf90"]
a_palette5 = ["#003049","#61988e","#f77f00","#7d1538","#8390fa"]
region_names =['gulf','labr','gdbk','nace','egrl']
region_longnames = ['Gulf Stream','Labrador Current','Grand Bank','NAC Extension','East Greenland Current']
region_longnames = dict(zip(region_names, region_longnames))
region_colors = dict(zip(region_names,a_palette5))

# rhs_list = ['e_ua','E','dif_h','dif_v','A','I','F']
rhs_list = ['A','F','dif_v','dif_h','E','e_ua','I']
term_colors = ['#fc8d62','#66c2a5','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494']
color_dic = dict(zip(rhs_list,term_colors))

error_color = 'r'

term_dic = {
    'A': r"$-u'\cdot \nabla \bar s$",
    'F': "Evap/Prec",
    'E': r"$-(u'\nabla s'-\overline{u'\nabla s'})$",
    # 'E': r"$\overline{u'\cdot \nabla s'}$",
    'dif_v': "Vertical Diffusion",
    'dif_h': "Horizontal diffusion",
    'e_ua': "Subdaily Advection",
    'I': "Surface salt flux and salt plume"
}
case_term_dic = {
    'A': "Anomalous Advection",
    'F': "Dilution",
    # 'E': r"$(-u'\nabla s'-\overline{u'\nabla s'})$",
    'E': "Rectified Advection",
    'dif_v': "Vertical Diffusion",
    'dif_h': "Horizontal diffusion",
    'e_ua': "NA Advection",
    'I': "External Heating",
    'Idif':"q'+vert diff"
}

# TOTAL_VOLUME_nep,NUMBER_OF_PARTICLE_nep,VOLUME_EACH_nep = (137447274971136.0, 9992, 13755731968.0)
TOTAL_VOLUME_nep,NUMBER_OF_PARTICLE_nep,VOLUME_EACH_nep = (296979741540352.0, 10053, 29541404709.07709)
TOTAL_VOLUME_wau,NUMBER_OF_PARTICLE_wau,VOLUME_EACH_wau = (63566417756160.0,10013, 6348388864.0)


fill_betweenx_kwarg = dict(
    color = 'grey',
    alpha = 0.5
)

from geopy.distance import geodesic

def calculate_projected_rectangle(lon1, lat1, lon2, lat2, perpendicular_distance, projection=ccrs.PlateCarree()):
    """
    Compute the four corners of a rectangle in a projected coordinate system.

    Parameters:
    lon1, lat1 : float : Longitude and Latitude of first point
    lon2, lat2 : float : Longitude and Latitude of second (adjacent) point
    perpendicular_distance : float : Distance along the perpendicular edge in projected space (km)
    projection : cartopy.crs : The map projection to use (default: PlateCarree)

    Returns:
    list : Four corner points [(lonA, latA), (lonB, latB), (lonC, latC), (lonD, latD)]
    """

    # Convert lat/lon to projected coordinates (meters)
    x1, y1 = projection.transform_point(lon1, lat1, ccrs.Geodetic())
    x2, y2 = projection.transform_point(lon2, lat2, ccrs.Geodetic())

    lat_test = lat1 + 1 / 111  # ~1 km in latitude
    x_test, y_test = projection.transform_point(lon1, lat_test, ccrs.Geodetic())
    
    # Compute how many projected units correspond to 1 km in this location
    meters_per_unit = np.hypot(x_test - x1, y_test - y1)

    # Convert perpendicular distance from km to projected coordinates
    perpendicular_distance_proj = perpendicular_distance * meters_per_unit

    # Compute direction vector along the given edge
    dx, dy = x2 - x1, y2 - y1

    # Compute perpendicular vector (-dy, dx) normalized
    length = np.hypot(dx, dy)
    for _ in range(7):
        perp_dx = (-dy / length) * perpendicular_distance_proj
        perp_dy = (dx / length) * perpendicular_distance_proj
    
        # Compute the four corners
        x3, y3 = x2 + perp_dx, y2 + perp_dy
        x4, y4 = x1 + perp_dx, y1 + perp_dy
    
        # Convert back to latitude/longitude
        normal_proj = ccrs.PlateCarree()
        lon3, lat3 = normal_proj.transform_point(x3, y3, projection)
        lon4, lat4 = normal_proj.transform_point(x4, y4, projection)
        
        actual_distance_km = geodesic((lat2, lon2), (lat3, lon3)).km
        adjustment_factor = abs(perpendicular_distance) / actual_distance_km if actual_distance_km > 0 else 1
        perpendicular_distance_proj *= adjustment_factor
        error = abs(actual_distance_km) - abs(perpendicular_distance)

        # perpendicular_distance_proj -= error*meters_per_unit
        print(f"Iteration: {_+1}, Target Distance: {perpendicular_distance:.4f} km, "
              f"Actual Distance: {actual_distance_km:.4f} km, Error: {error:.4f} km")

    return np.array([(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4)])

# Example usage
lon1, lat1 = -120, 39.5  # Example point
lon2, lat2 = -102.5, 22.5  # Another point (slightly east)
perpendicular_distance = -2000  # 10 km perpendicular edge

rectangle_corners = calculate_projected_rectangle(lon1, lat1, lon2, lat2, perpendicular_distance,projection = projection)

print("Rectangle Corners:", rectangle_corners)
[(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4)] = rectangle_corners