import numpy as np
import matplotlib.pyplot as plt
import math

#function parameters
GM = 4902.8001e9 #gravitational constant
R_moon = 1737e3 #radius
h = 100e3 #altitude
a = R_moon + h #semimajor axis
e = 0.01 #eccentricity
i = 45 #inclination
w = 0 #argument of pericenter (pericenter is the same as periapsis)
W = 0 #longitude
M0 = 0 #mean anomaly at time t0
t0 = 0 #initial time
period = 1*86400 
t = np.linspace(t0, t0 + period, 10000)


def elem2coord(a, e, i, w, W, M0, t0, t, m):

    #% INPUTS
    #% a - semimajor axis
    #% e - eccentricity
    #% i - inclination
    #% w - argument of pericenter (pericenter is the same as periapsis)
    #% W - longitude of the accending nodes
    #% M0 - mean anomaly at time t0
    #% t0 - initial time
    #% t  - time
    #% m  - gravitational constant (GM)

    #% OUTPUT
    #% x,y,z - position
    #% X,Y,Z - velocity


    
    i = i / 180 * np.pi
    w = w / 180 * np.pi
    W = W / 180 * np.pi
    M0 = M0 / 180 * np.pi

    n = np.sqrt(m / a / a / a)
    M = M0 + n * (t - t0)

    d = np.ones(M.shape)
    E = M + e * np.sin(M)
    eps = 1e-20

    while (d > eps).all():
        E_1 = E
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        d = np.abs(E - E_1)

    v = np.sqrt((1 + e) / (1 - e)) 
    v = 2 * np.arctan(v * np.tan(E / 2))

    u = w + v

    ax = np.cos(W) * np.cos(u) - np.sin(W) * np.sin(u) * np.cos(i)
    ay = np.sin(W) * np.cos(u) + np.cos(W) * np.sin(u) * np.cos(i)
    az = np.sin(u) * np.sin(i)
    r = a * (1 - e * np.cos(E))
    x = r * ax
    y = r * ay
    z = r * az

    axu = -np.cos(W) * np.sin(u) - np.sin(W) * np.cos(u) * np.cos(i)
    ayu = -np.sin(W) * np.sin(u) + np.cos(W) * np.cos(u) * np.cos(i)
    azu = np.cos(u) * np.sin(i)

    p = a * (1 - e * e)

    X = np.sqrt(m / p) * ((e * np.sin(v)) * ax + (1 + e * np.cos(v)) * axu)
    Y = np.sqrt(m / p) * ((e * np.sin(v)) * ay + (1 + e * np.cos(v)) * ayu)
    Z = np.sqrt(m / p) * ((e * np.sin(v)) * az + (1 + e * np.cos(v)) * azu)

    return x, y, z, X, Y, Z

x, y, z, X, Y, Z = elem2coord(a, e, i, w, W, M0, t0, t, GM)

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
xM = R_moon * np.cos(u)*np.sin(v)
yM = R_moon * np.sin(u)*np.sin(v)
zM = R_moon * np.cos(v)

ax = plt.axes(projection='3d')
ax.plot_wireframe(xM/1000, yM/1000, zM/1000, color="gray")
ax.plot3D(x/1000,y/1000,z/1000)



#plt.plot(x,y)


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

    #ax+by+cz = d | d = a**2 + b**2 + c**2

    #dot loc of satelltie
    #if ax+by+cz > d
    #if ax+by+cz < d

#def constellation(num_sat):

    



    
    
    
    
    
