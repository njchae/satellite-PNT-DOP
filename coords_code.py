import numpy as np
import math as math
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
import os as os

fig,ax = plt.subplots()
#function parameters
GM = 4902.8001e9 #gravitational constant
R_moon1 = 1737e3 #radius
h1 = 500e3 #altitude
a1 = R_moon1 + h1 #semimajor axis
e1 = .001 #eccentricity
i1 = 120 #inclination # 0 to 180
wp1 = 0 #argument of pericenter (pericenter is the same as periapsis)#can change but doesnt matter much if ececntircity 
w1 = 30 #longitude#can change
M01 = 0 #mean anomaly at time t0#where i put the satellite where i am in the orbit 0- 360
t01 = 0 #initial timeyes
period1 = 1*86400 

tint = 2 #time intervals

t1 = np.linspace(t01,t01 + period1, tint)






def elem2coord(a=a1,e=e1,i=i1,w=wp1,W=w1,M0=M01,t0=t01,t=t1,m=GM):

    #% INPUTS
    #% a - semimajor axis
    #% e - eccentricity
    #% i - inclination
    #% w - argument of pericenter (pericenter is the same as periapsis)
    #% W - longitude of the accending nodes
    #% M0 - mean anomaly at time t  0
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

    VX = np.sqrt(m / p) * ((e * np.sin(v)) * ax + (1 + e * np.cos(v)) * axu)
    VY = np.sqrt(m / p) * ((e * np.sin(v)) * ay + (1 + e * np.cos(v)) * ayu)
    VZ = np.sqrt(m / p) * ((e * np.sin(v)) * az + (1 + e * np.cos(v)) * azu)

    return [x, y, z]
#return all instances

def constellation(numorbits, numsats, typeorbit=0):
    #need to implement typeorbit which makes this function super complicated to accomodate every possible practical orbital pattern
    if typeorbit == 0: # this is orbits distributed even among 6 vertical orbits across the the angles of the longitude
        currmean = 0
        currlong = 0
        longdiv = numorbits//numsats
        satarr = []


        orbadd = 360/numorbits
        satadd = 360/(numsats/numorbits)/2
        counter = 0
        while currlong < 180:
            currmean = 0
            while currmean < 360:
                newsat = elem2coord(W=currlong, M0=currmean)
                satarr.append(newsat)
                currmean += satadd

            currlong += orbadd
    #can have more types of orbits to implement!
    
    return satarr

sat_loc = ([elem2coord(W=0, M0=0),elem2coord(W=0, M0=45),elem2coord(W=0, M0=90),elem2coord(W=0, M0=135),elem2coord(W=0, M0=180),elem2coord(W=0, M0=225),elem2coord(W=0, M0=270),elem2coord(W=0, M0=315)])
sat_loc.extend([elem2coord(W=90, M0=0),elem2coord(W=90, M0=45),elem2coord(W=90, M0=90),elem2coord(W=90, M0=135),elem2coord(W=90, M0=180),elem2coord(W=90, M0=225),elem2coord(W=90, M0=270),elem2coord(W=90, M0=315)])
sat_loc.extend([elem2coord(W=270, M0=0),elem2coord(W=270, M0=45),elem2coord(W=270, M0=90),elem2coord(W=270, M0=135),elem2coord(W=270, M0=180),elem2coord(W=270, M0=225),elem2coord(W=270, M0=270),elem2coord(W=270, M0=315)])
sat_loc.extend([elem2coord(W=180, M0=0),elem2coord(W=180, M0=45),elem2coord(W=180, M0=90),elem2coord(W=180, M0=135),elem2coord(W=180, M0=180),elem2coord(W=180, M0=225),elem2coord(W=180, M0=270),elem2coord(W=180, M0=315)])


def visibility(sat_loc):
    #sat_loc is an array of satellite xyz coords
    
    counter = 0
    count = []
    visiondict = []
    time = t01
    tindex = 0
    for lat in range(-90,91,1):
        print(lat)
        for lon in range(-180,181,1):

            lon1 = lon / 180 * np.pi
            lat1 = lat / 180 * np.pi
            R = 1737 * 1000 #m 
            x_1 = R * np.cos(lat1) * np.cos(lon1)
            y_1 = R * np.cos(lat1) * np.sin(lon1)
            z_1 = R * np.sin(lat1)
        
            d = x_1 ** 2 + y_1 ** 2 + z_1 ** 2
            time = 0
            for tindex in range(tint):
                # format of [lat,lon,time,how many sats,which sats]
                templist = [lat,lon,time]
                sat_vis = 0
                count = []
                satind = 0
                for sat in sat_loc:
                    sat_plane = x_1 * sat[0][tindex] + y_1 * sat[1][tindex] + z_1 * sat[2][tindex]
                   
                    if  sat_plane > d:
                        count.append(satind)
                    satind += 1
                sat_vis = len(count)
                templist.extend([sat_vis, count])
               
                visiondict.append(templist)
                time += period1/tint

    return visiondict

def get_R(sat, rec):
    return np.sqrt((rec[0] - sat[0]) ** 2 + (rec[1] - sat[1]) ** 2 + (rec[2] - sat[2]) ** 2)

def DOP(sat_loc, doptype):
    #sat_loc is an array of satellite xyz coords
    
    counter = 0
    count = []
    DOPdict = []
    time = t01
    tindex = 0
    #iterate through lat lon
    for lat in range(-90,91,1):
        for lon in range(-180,181,1):
            
                
            lon1 = lon / 180 * np.pi
            lat1 = lat / 180 * np.pi
            R = 1737 * 1000 #m 
            x_1 = R * np.cos(lat1) * np.cos(lon1)
            y_1 = R * np.cos(lat1) * np.sin(lon1)
            z_1 = R * np.sin(lat1)
            if lat == -60 and lon == 120:
               print(x_1,y_1,z_1)
        
            d = x_1 ** 2 + y_1 ** 2 + z_1 ** 2
            time = 0
            tempcoords = [x_1,y_1,z_1]
            for tindex in range(tint):
                templist = [lat,lon,time]
                sat_vis = 0
                count = []
                satind = 0
                # format of [lat,lon,time,DOP,typeofDOP]
                if doptype == 'PDOP':
                    templist.extend([DOP_calc(sat_loc, tempcoords,'PDOP',tindex),'PDOP'])
                elif doptype == 'TDOP':
                    templist.extend([DOP_calc(sat_loc, tempcoords, 'TDOP', tindex),'TDOP'])
                elif doptype == 'GDOP':
                    templist.extend([DOP_calc(sat_loc, tempcoords, 'GDOP',tindex),'GDOP'])
                DOPdict.append(templist)
                time += period1/tint
    return DOPdict

def DOP_calc(sat_coords, rec_coords,doptype, interval):
    #Dilution of Precision
    #inputs are nested lists of xyz coords
    rec_x = rec_coords[0]
    rec_y = rec_coords[1]
    rec_z = rec_coords[2]
    A = []
    #generate array of nested lists, each item corresponding to a row in the matrix
    for i in sat_coords:
        x = rec_x - i[0][interval]
        y = rec_y - i[1][interval]
        z = rec_z - i[2][interval]
        
        R = get_R(i, rec_coords, interval)
        
        A.append([x / R, y / R, z / R, 1])

    A = np.array(A)
    #print("A matrix \n", A, "\n")
    #print("A transpose \n", A.transpose(), "\n")
    A_T = A.transpose()

    #Q is covariance matrix
    Q = np.linalg.inv(np.dot(A_T, A))
    #print("Q matrix \n", Q, "\n")
    
    if rec_coords == [-434249.99999999994, 752143.0631867852, -1504286.12637357]:
        print('firstq')
        print(Q)
        print(Q[0, 0] + Q[1, 1] + Q[2, 2])
        print("Q[0, 0]: ", Q[0, 0], " Q[1, 1]: ", Q[1, 1], " Q[2, 2]: ", Q[2, 2])
    PDOP = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
    TDOP = np.sqrt(Q[3, 3])
    GDOP = np.sqrt(PDOP ** 2 + TDOP ** 2)
    if doptype == 'PDOP':
        return PDOP
    elif doptype == 'TDOP':
        return TDOP
    elif doptype == 'GDOP':
        return GDOP


def get_R(sat, rec, interval):
    return np.sqrt((rec[0] - sat[0][interval]) ** 2 + (rec[1] - sat[1][interval]) ** 2 + (rec[2] - sat[2][interval]) ** 2)


def cart2sph(x,y,z):
    hypotxy = np.sqrt(x*x+y*y)
    lat = np.arctan2(z, hypotxy)
    lon = np.arctan2(y,x)
    return [[-lat*180/np.pi],[lon*180/np.pi]]
#sat_loc = constellation(6,24)
#sat_loc.extend([elem2coord(W=0, M0=0),elem2coord(W=0, M0=22.5),elem2coord(W=0, M0=45),elem2coord(W=0, M0=67.5),elem2coord(W=0, M0=90),elem2coord(W=0, M0=112.5),elem2coord(W=0, M0=135),elem2coord(W=0, M0=157.5),elem2coord(W=0, M0=180),elem2coord(W=0, M0=202.5),elem2coord(W=0, M0=225),elem2coord(W=0, M0=247.5),elem2coord(W=0, M0=270),elem2coord(W=0, M0=292.5),elem2coord(W=0, M0=315),elem2coord(W=0, M0=337.5)])


#sat_loc = ([elem2coord(M0=0),elem2coord(M0=90), elem2coord(M0=180),elem2coord(M0=270)])
#print(sat_loc)
hold = visibility(sat_loc)
def dataget(frame):
    z = np.zeros([181, 361])
    w = 0
    for z2 in list(range(0,181)[::-1]):
        for z1 in list(range(0,361)):
            z[z2, z1] = hold[w+frame][3]
            w += tint
    return 

def animator(frame):
   
    ax.clear()
    plt.xlabel("Longitude in degrees")
    plt.ylabel("Latitude in degrees")   
   
    z = dataget(frame)

    y = range(-90,91,1)
    x = range(-180,181,1)
    ax.pcolor(x,y,z,cmap='Reds')
    ax.set_title("time at interval: " +str(frame))
    
    for i in sat_loc:
        holder1 = cart2sph(i[0][frame],i[1][frame],i[2][frame])
        ax.plot(holder1[1], holder1[0], marker="+", markersize=20, markeredgecolor="green", markerfacecolor="green")
        
    plt.show()

fig_folder ="C:\\Users\henry\OneDrive\Desktop\heatmaps"

def plotter(satloc, numtime):    
    plt.xlabel("Longitude in degrees")
    plt.ylabel("Latitude in degrees")   
    y = range(-90,91,1)
    x = range(-180,181,1)
    z = np.zeros([181, 361])
    w = 0
    for z2 in list(range(0,181)[::-1]):
        for z1 in list(range(0,361)):
            z[z2, z1] = hold[w+0][3]
     
            w += tint

    plt.pcolor(x,y,z,cmap='Reds')
    c = plt.colorbar(label = "# of satellites visible")
    tempanim = animation.FuncAnimation(fig,animator,frames = tint,blit = False,interval=10)
        #plt.savefig(fig_folder  + "\\anim" + str(tc) + ".png",dpi=400)
    plt.show()

plotter(sat_loc,tint)

#print([elem2coord(W=0, M0=0),elem2coord(W=0, M0=22.5),elem2coord(W=0, M0=45),elem2coord(W=0, M0=67.5),elem2coord(W=0, M0=90),elem2coord(W=0, M0=112.5),elem2coord(W=0, M0=135),elem2coord(W=0, M0=157.5),elem2coord(W=0, M0=180),elem2coord(W=0, M0=202.5),elem2coord(W=0, M0=225),elem2coord(W=0, M0=247.5),elem2coord(W=0, M0=270),elem2coord(W=0, M0=292.5),elem2coord(W=0, M0=315),elem2coord(W=0, M0=337.5)])
#instances of time for locatinos as wlel 
#lat lon 
#bunch of inputs
#numver of satellites visible
#dictionary
#answer questions
#fractions min max  etx, gaps
