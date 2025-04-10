import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['figure.facecolor'] = 'white'

from NuRadioReco.utilities import units

import pykonal
import os
import _pickle as cpk
from scipy import interpolate
from datetime import datetime as dt

cdict = {'red': ((0.0, 0.0, 0.0),(0.1, 0.5, 0.5),(0.2, 0.0, 0.0),(0.4, 0.2, 0.2),(0.6, 0.0, 0.0),(0.8, 1.0, 1.0),(1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),(0.1, 0.0, 0.0),(0.2, 0.0, 0.0),(0.4, 1.0, 1.0),(0.6, 1.0, 1.0),(0.8, 1.0, 1.0),(1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),(0.1, 0.5, 0.5),(0.2, 1.0, 1.0),(0.4, 1.0, 1.0),(0.6, 0.0, 0.0),(0.8, 0.0, 0.0),(1.0, 0.0, 0.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)



#=====#GEOMETRY#=====#

def Ind(arr,val):
    #when querying a position on the map, you interact with the map through the indices not the axes directly (distance & depth arrays)
    #>Input: array of distances/depths and distance/depth of chosen position
    #>Output: Index of closest value in distance/depth array given the step size
    return np.argmin(np.abs(arr-val))

#Note to self: MAKE SURE THIS IS THE CORRECT FORMAT
def getCardinalDistance(pos,pulser_pos):
    #
    return  np.sqrt(pulser_pos[0]**2+pulser_pos[1]**2) - np.sqrt((pulser_pos[0]-pos[0])**2+(pulser_pos[1]-pos[1])**2)

def get_projectedCoords(pos,pulser_pos): # x,y,z => r,z
    return np.round(getCardinalDistance(pos,pulser_pos),2) , pos[-1]



#======#ICE MODELS#======#
def iceModelPoly5(z, fitParameters):
    z0 = 75.0
    Pz = 0
    C = 0.8506*(units.cm**3/units.gram)*(6.241509744511525e+36)
    for ia,a in enumerate(fitParameters):
        Pz+=a*np.exp(ia*z/z0)
    return 1 + C * Pz 

def iceModelExp3(z,fitParameters):
    av = fitParameters
    C = 0.8506*(units.cm**3/units.gram)*(6.241509744511525e+36)
    #z_sf = -10. #snow-firn boundary depth
    #z_fb = -65. #firn-bubble ice boundary depth
    z_sf = -9.54
    z_fb = -62.6

    p_bulk = av[0]
    p_snow = (av[1]*np.exp(av[2]*-1*(z+av[3]))) * (z > z_sf)
    p_firn = (av[4]*np.exp(av[5]*-1*(z+av[6]))) * (z <= z_sf) * (z > z_fb)
    p_bubl = (av[7]*np.exp(av[8]*-1*(z+av[9]))) * (z <= z_fb)

    return 1+ C * ( p_bulk + p_snow + p_firn + p_bubl )


#=======# GENERATING TIME MAP #=======#
def generateTimeMap(V_Matrix,xrange,yrange,emitter_ind):
    start_t = dt.now()
    stepx = np.abs(xrange[1]-xrange[0])
    stepy = np.abs(yrange[1]-yrange[0])

    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = xrange[0], yrange[0], 0

    solver.velocity.node_intervals = stepx,stepy, 1
    solver.velocity.npts = len(xrange), len(yrange),1
    solver.velocity.values = V_Matrix

    src_idx = tuple(emitter_ind)
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False

    solver.trial.push(*src_idx)
    solver.solve()
    print(f'This map took {(dt.now()-start_t).seconds} seconds to generate.')

    return solver.traveltime.values.T[0]



def getTime(tmap,xdom,ydom,position):
    #get time from map, translating position from meters to map indices
    #> Input: position in meters
    #> Output: arrival time in ns
    return tmap[Ind(ydom,position[1]),Ind(xdom,position[0])]#a bit confusing, the map is indexed [z,r], not [r,z]


def get_InterpolatedTime(sourceDepth,sourceDepthStep,time_Maps,rdomain,zdomain,position):
    #estimate an arrival time for a pulser depth between existing maps

    sourceDepth_range = [float(sourceDepth-(sourceDepth%sourceDepthStep)),float(sourceDepth-(sourceDepth%sourceDepthStep)+sourceDepthStep)]
    print(f'There is no pre-generated map for {sourceDepth} m, instead will interpolate between {sourceDepth_range} m')

    #get time at position (r,z) for both boundary maps
    t_range=[]
    for tm in time_Maps:
        t_range.append(getTime(tm,rdomain,zdomain,position))

    #interpolate time values between map values
    depth_domain = np.array([np.round(x,3) for x in np.arange(sourceDepth_range[0],sourceDepth_range[-1],0.01)])
    fin = interpolate.interp1d(sourceDepth_range, t_range)

    return fin(depth_domain)[list(depth_domain).index(np.round(sourceDepth,3))]



#=======# PLOTTING #=======#

def plotTmap(TMap,rdomain,zdomain,station):
    rlim = [int(rdomain[0]),int(rdomain[-1])]
    zlim = [int(zdomain[0]),int(zdomain[-1])]
    fig, ax = plt.subplots()

    source_loc = np.array(np.where(TMap==np.min(TMap))).T[0][::-1]
    im1 = ax.imshow(TMap,origin='upper',cmap=my_cmap, vmin=np.min(TMap), vmax=2000,aspect = 'auto')
    cbar = fig.colorbar(im1)
    cbar.set_label('Travel Time [ns]', labelpad=15, rotation=270)

    xlabs = np.array([  rdomain[0],  rdomain[int(len(rdomain)/2)],  rdomain[-1]])
    xta = [list(rdomain).index(float(x)) for x in xlabs]
    xlabs = [int(x) for x in xlabs]
    plt.xticks(xta,xlabs)
    
    ylabs = np.array([  zdomain[0],  zdomain[int(len(zdomain)/2)] ,  zdomain[-1] ])
    yta = [list(zdomain).index(y) for y in ylabs]
    ylabs = [int(x) for x in ylabs]
    plt.yticks(yta,ylabs)
    
    plt.scatter(source_loc[0],source_loc[1],c='m',marker='x',label=f'Source ({np.round(rdomain[source_loc[0]],2)}, {np.round(zdomain[source_loc[1]],2)}) m')
    plt.xlabel('r [m]')
    plt.ylabel('z [m]')

    plt.suptitle(f'Station {station} | \n r: {rlim} m | z: {zlim} m')
    plt.legend(bbox_to_anchor=(1.25,1),loc='upper left')
    plt.show()


#=======# SAVING & LOADING PREGENERATED MAPS TO/FROM pkl FILES #=======#

def saveTMap(TMap,sourceDepth,rdomain,zdomain,station,map_dirpath):
    label=str(float(abs(sourceDepth))).replace('.','p')
    map_path =map_dirpath + f'propagation_map_s{station}_{label}.pkl'

    if os.path.isfile(map_path):
        with open(map_path, 'rb') as pf:
            output_dict = cpk.load(pf)
    else:
        output_dict = {}
    if str(float(sourceDepth)) not in output_dict:
        output_dict[str(float(sourceDepth))] = {}
    output_dict[str(float(sourceDepth))] = TMap

    with open(map_path, 'wb') as pfs:
        cpk.dump(output_dict, pfs)



def loadTMaps(station,sourceDepth,sourceDepthStep,map_dirpath):
    #sourceDepth => depth of pulser
    #sourceDepthStep => step size between generated maps

    tmap_list =[]
    if sourceDepth%sourceDepthStep == 0: #if exact map exists with specified source depth
        tmap_list.append(loadSingleMap(station,sourceDepth,map_dirpath)) #get single map

    else: #if specified depth is between the source depth of two adjacent maps
        tmap_list = loadBorderMaps(station,sourceDepth,sourceDepthStep,map_dirpath) #get two maps on either end
    return np.array(tmap_list)



def loadSingleMap(station,sourceDepth,map_dirpath):
    label=str(float(abs(sourceDepth))).replace('.','p')
    with open(map_dirpath+f'propagation_map_s{station}_{label}.pkl', 'rb') as file:
        return cpk.load(file)['station_'+str(station)]


def loadBorderMaps(station,sourceDepth,sourceDepthStep,map_dirpath):
    
    existingDepths = [sourceDepth-(sourceDepth%sourceDepthStep), sourceDepth-(sourceDepth%sourceDepthStep)+sourceDepthStep]
    #e.g. if sourceDepth=12.3 m & step = 1 m, this would be [12,13]

    maplist=[]
    for depthVal in existingDepths:
        label=str(float(abs(depthVal))).replace('.','p')
        with open(map_dirpath+f'propagation_map_s{station}_{label}.pkl', 'rb') as file:
            maplist.append(cpk.load(file)[str(float(depthVal))])
    return np.array(maplist)















