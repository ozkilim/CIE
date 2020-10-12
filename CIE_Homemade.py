# CIE GENERATION SYSTEM
import math
import matplotlib
import matplotlib.pyplot
from kapteyn import maputils
from numpy import arange
from matplotlib import pyplot as plt
from kapteyn import maputils
from service import *
import numpy as np
import matplotlib.pyplot as plt
from pyephem_sunpath.sunpath import sunpos
from datetime import datetime
import matplotlib.cm as cm
from matplotlib.pylab import meshgrid
from matplotlib.pylab import cbook

def peremeterInitialiser(skyType):    
    # List of a,b,c,d,e perameters for each sky type
    skyPeramDataSet = [[4,-0.7,0,-1,0],[4, -0.7 ,2, -1.5, 0.15],
    [1.1, -0.8, 0,-1, 0],[1.1, -0.8, 2, -1.5, 0.15] ,[0, -1,0, -1, 0],
    [0, -1, 2, -1.5, 0.15] ,[0, -1, 5, -2.5, 0.30],
    [0, -1, 10, -3, 0.45],[-1, -0.55, 2, -1.5, 0.15],
    [-1, -0.55, 5, -2.5, 0.30] ,[-1,0 -0.55, 10, -3, 0.45],
    [-1, -0.32, 10, -3, 0.45] ,[-1, -0.32, 16, -3, 0.30] ,
    [-1, -0.15, 16, -3, 0.30],[-1, -0.15, 24, -2.8, 0.15]]
    # get the correct a,b,c,d,e for the selected sky type
    skyPerameters = skyPeramDataSet[(skyType-1)]

    # List of A1,A2,B,C,D,E perameters for each sky type
    zenithLumPeramsDataSet = [[0.957, 1.790, 21.72, 4.52, 0.64, 34.56],[0.830, 2.030, 29.35 ,4.94, 0.70, 30.41]
    ,[0.600, 1.500, 10.34, 3.45, 0.50, 27.47],[0.567, 2.610, 18.41, 4.27, 0.63, 24.04]
    ,[1.440, -0.750, 24.41, 4.60, 0.72, 20.76],[1.036, 0.710, 23.00, 4.43, 0.74, 18.52]
    ,[1.244, -0.840, 27.45, 4.61, 0.76, 16.59],[0.881, 0.453, 25.54, 4.40, 0.79, 14.56]
    ,[0.418, 1.950, 28.08, 4.13, 0.79, 13.00]]
    # onny 7 -  15 here...
    zenithLumPerams = zenithLumPeramsDataSet[(skyType-7)]

    return skyPerameters,zenithLumPerams

def solarPosition(year, month, day, hour, timezone, lat, lon):
    # Get sunAlt, sunAzm geometry from the time dat location ect..
    thetime = datetime(year, month, day, hour)
    sunAlt, sunAzm = sunpos(thetime, lat, lon, timezone, dst=False)
    # thse are the bratislava coordiantes for comparason uncomment to compare to paper
    # "STANDARD SKY CALCULATIONS FOR DAYLIGHTING DESIGN AND ENERGY PERFORMANCE PURPOSES"..
    # sunAlt = 38.02
    # sunAzm = 147.67

    return sunAlt, sunAzm

def φ(Z, a, b):
    # The luminance gradation function φ relates the luminance of a sky element to its zenith angle:
    φ = 1 + a*math.exp(b/math.cos(Z))

    return(φ)

def f(x, c, d, e):
    f = 1+c*(math.exp(d*x)-math.exp(d*math.pi/2))+e*((math.cos(x))**2)
    
    return(f)

def Lγα(Z, α, sunAlt, sunAzm,skyPerameters):
    Zs = math.radians(90 - sunAlt)
    Z = math.radians(Z)

    a = skyPerameters[0]
    b = skyPerameters[1]
    c = skyPerameters[2]
    d = skyPerameters[3]
    e = skyPerameters[4]

    Az = math.radians(abs(α-sunAzm))
    x = math.acos((math.cos(Zs)*math.cos(Z)) +
                  (math.sin(Zs)*math.sin(Z)*math.cos(Az)))
    zenith_Norm = f(x, c, d, e)*φ(Z, a, b)/(f(Zs, c, d, e)*φ(0, a, b))
    LγαRatio = zenith_Norm
    # this is the ratio of Lγα/Lz
    return(LγαRatio)


def calcLumTurbFactor(skyType):
    # Caluculate luminous turbidity factor Tv
    # Tv may need to be calculated for each sky type for best reasults.. need function to generalise this...
    # If global illuminance Gv and diffuse illuminance
    # Dv are measured, ...approx or from weather...
    # Evs is the direct-beam illuminance on a horizontal surface (at the ground);
    # Evs = from reaeal sky data..
    # Evs = 
    # # Ev is the extraterrestrial horizontal illuminance;
    # Ev =  133.8*math.sin(oneSunAlt)
    # # m is the relative optical air mass. 
    # # check where magic numbers come from.
    # m = 1/(math.sin(oneSunAlt) + 0.50572*(oneSunAlt +6.07995)**(-1.6364))
    # print(m)
    # # av is the luminous extinction coefficient;
    # av = 1/(9.9+0.043*m)
    # print(m)
    # Tv  = -(np.log(Evs/Ev))/(av*m)

    # if no weather data... use approximation..
    # approximate values for Tv given in "CIE GENERAL SKY STANDARD DEFINING LUMINANCE DISTRIBUTIONS "
    TvApprox = [45, 20,45,20,45,20,12,10,12,10,4,2.5,4.5,5,4]
    Tv = TvApprox[skyType-1]

    return Tv

def getLz(oneSunAlt,zenithLumPerams,skyType):
    # This calculates th zenith luminance in for th model to give absolute values out

    # Works in degrees.Its a polynomial approxiamtion from another model.
    # Lz = -0.65382+((0.10065)*oneSunAlt)-((0.00380)*(oneSunAlt**2))+((7.95867*np.exp(-5)) *
                                                        # (oneSunAlt**3)) - ((7.99933*np.exp(-7))*(oneSunAlt**4)) + ((3.21145*np.exp(-9))*(oneSunAlt**5))
    # turn degrees to radians.                                                    
    oneSunAlt = math.radians(oneSunAlt)

    # get the zenithLumPerams for the correct sky type
    A1 = zenithLumPerams[0]
    A2 = zenithLumPerams[1]
    B = zenithLumPerams[2]
    C = zenithLumPerams[3]
    D = zenithLumPerams[4]
    E = zenithLumPerams[5]

    Tv = calcLumTurbFactor(skyType)
    # Tv = 4.5

    A = (A1*Tv) +A2
    Lz = A*math.sin(oneSunAlt) + ((0.7*(Tv+1)*(math.sin(oneSunAlt)
                                                  ** C))/(((math.cos(oneSunAlt))**D))) + 0.004*Tv
    return Lz*1000

def generateLuminances(sunAlt, sunAzm,zenithLumPerams,skyPerameters,skyType):
    # Main caluclation 
    Lz = getLz(sunAlt,zenithLumPerams,skyType)
    FNorm = 0
    n = 360*90
    phispace = np.linspace(0, 2*math.pi, 100)
    thetaspace = np.linspace(-math.pi/2, math.pi/2, 100)
    dtheta = thetaspace[3] - thetaspace[2]
    dphi = phispace[3] - phispace[2]

    areaNorm = 0
    for theta in thetaspace:
        for phi in phispace:
            # print(dphi*abs((math.cos(theta) - math.cos(theta + dtheta))))
            areaNorm = areaNorm + dphi * \
                abs((math.cos(theta) - math.cos(theta + dtheta)))
    finalAreaNorm = areaNorm/2

    areaMatrix = np.zeros((90, 360))
    LγαRatioMatrix = np.zeros((90, 360))
    for α in range(0, 360):
        for Z in range(0, 90):
            # record value for Z = 0 to use in normalisation
            if Z == 0:
                Lzen = Lγα(Z, α, sunAlt, sunAzm,skyPerameters)
            else:
                LγαRatioMatrix[Z, α] = Lγα(Z, α, sunAlt, sunAzm,skyPerameters)

    # multiply ratio by the zenith luminance to get aboslute luminance for every az and el
    lumDist = LγαRatioMatrix*Lz

    return lumDist

def showSky(NormalisedLum):
    # Plot the sky out for user
    a = np.linspace(0, 2*np.pi, 360)
    b = np.linspace(0, 1, 90)
    # actual plotting
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_yticklabels([])
    clev = np.arange(NormalisedLum.min(),NormalisedLum.max(),1000) #Adjust the .001 to get finer gradient
    ctf = ax.contourf(a, b, NormalisedLum,clev, cmap=cm.jet)
    plt.colorbar(ctf)
    plt.show()

def Main(year, month, day, hour, timezone, lat, lon,skyType):
    # Main function to generate simulation
    skyPerameters,zenithLumPerams = peremeterInitialiser(skyType)
    sunAlt, sunAzm = solarPosition(year, month, day, hour, timezone, lat, lon)
    lumDist = generateLuminances(sunAlt, sunAzm,zenithLumPerams,skyPerameters,skyType)
    showSky(lumDist)

Main(2020, 5, 1, 9, 1, 48.1486, 17.1077,11)