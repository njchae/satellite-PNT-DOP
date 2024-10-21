import numpy as np

def visibility(lat, lon, sat_loc):
    #sat_loc is an array of satellite xyz coords
    R = 1737 * 1000 #m
    x_1 = R * np.cos(lat) * np.cos(lon)
    y_1 = R * np.cos(lat) * np.sin(lon)
    z_1 = R * np.sin(lat)

    point_0 = (0, 0, 0)
    point_1 = (x_1, y_1, z_1)

    d = x_1 ** 2 + y_1 ** 2 + z_1 ** 2

    count = []
    for sat in sat_loc:
        sat_plane = x_1 * sat[0] + y_1 * sat[1] + z_1 * sat[2]
        if  sat_plane > d:
            count = count + [sat]
    sat_vis = len(count) 
    return sat_vis