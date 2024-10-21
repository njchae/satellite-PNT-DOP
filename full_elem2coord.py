import numpy as np
import matplotlib.pyplot as plt
 #function parameters
GM = 4902.8001e9 #gravitational constant
R_moon1 = 1737e3 #radius
h1 = 100e3 #altitude
a1 = R_moon1 + h1 #semimajor axis
e1 = 0.01 #eccentricity
i1 = 90 #inclination
wp1 = 0 #argument of pericenter (pericenter is the same as periapsis)
w1 = 0 #longitude
M01 = 0 #mean anomaly at time t0
t01 = 0 #initial time
period1 = 1*86400
t1 = np.linspace(t01,t01 + period1, 100)
def elem2coord(a=a1,e=e1,i=i1,wp=w1,w=w1,M0=M01,t0=t01,t=t1,m=GM):
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
    i = i/180*np.pi
    w = w/180*np.pi
    W = W/180*np.pi
    M0 = M0/180*np.pi
    n = np.sqrt(m / a / a / a)
    M = M0 + n * (t - t0)
    d = 1
    E = M + e * np.sin(M)
    eps = 1e-20
    while d > eps:
        #%E1=E E=M+e*np.sin(E)
        #%d=abs(E-E1)
        E1 = E
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        d = abs(E - E1)
    v = np.sqrt((1 + e) / (1 - e))
    v = 2 * np.atan(v * np.tan(E / 2))
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
    return [x, y, z, X, Y, Z]

default = 0
def constellation(numorbits, numsats, typeorbit=default):
    #need to implement typeorbit which makes this function super complicated to accomodate every possible practical orbital pattern
    currinc = 0
    currlong = 0
    longdiv = numorbits//numsats
    satarr = []
    orbadd = 360/numorbits
    satadd = 360/numsats
    while currlong <= 360:
        while currinc <= 360:
             newsat = elem2coord(w=currlong, i=currinc)
             satarr.append(newsat)
             currinc += satadd
        currlong += orbadd
    return satarr

def totalvis(numorbits, numsats, typeorbit=default):
    sat_loc = constellation(numorbits, numsats)
    print("The minimum visibilty at any point on the surface is ", minvis(sat_loc))
    polvis1 = polvis(sat_loc)
    print("Visibility at the the northpole is ", polvis1[0], "while the south pole is ", polvis1[1])
    return

def polvis(sat_loc):
    #
    lon = 0
    lat = 90
    visarr = []
    visarr.append(visibility(lat,lon,sat_loc))
    lat = -90
    visarr.append(visibility(lat,lon,sat_loc))
    return visarr

def minvis(numorbits, numsats, sat_loc):
    #lets us know the min vision
    lon = -180
    lat = -90
    currminvis = 0
    while lon <= 180:
        while lat <= 90:
            currminvis = min(currminvis, visibility(lat, lon, sat_loc))
            lat += 1
        lon += 1
    return minvis

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
    return count

    #create function that is a map of visibility
    #3 arrays: latitude, longitude, visibilty

    #Lon = np.linspace(0,2*pi,100)
    #lat= np.linspace(-pi/2,pi/2,100)
    #Lon,lat = np.meshgrid(lon,lat)

    #num_sat_visible function

    #num_sat_visible = visibility(… lon, lat)
        #This would create a 2D numpy array for visibility

    #Then you do plt.pcolor(Lon ,lat, num_sat_visible) to plot the map

    # Minvis = np.min(num_sat_visible)
