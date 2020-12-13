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
    skyPeramDataSet = [[4, -0.7, 0, -1, 0], [4, -0.7, 2, -1.5, 0.15],
                       [1.1, -0.8, 0, -1, 0], [1.1, -0.8,
                                               2, -1.5, 0.15], [0, -1, 0, -1, 0],
                       [0, -1, 2, -1.5, 0.15], [0, -1, 5, -2.5, 0.30],
                       [0, -1, 10, -3, 0.45], [-1, -0.55, 2, -1.5, 0.15],
                       [-1, -0.55, 5, -2.5, 0.30], [-1, -0.55, 10, -3, 0.45],
                       [-1, -0.32, 10, -3, 0.45], [-1, -0.32, 16, -3, 0.30],
                       [-1, -0.15, 16, -3, 0.30], [-1, -0.15, 24, -2.8, 0.15]]
    # get the correct a,b,c,d,e for the selected sky type
    skyPerameters = skyPeramDataSet[(skyType-1)]

    # List of A1,A2,B,C,D,E perameters for each sky type
    zenithLumPeramsDataSet = [[0.957, 1.790, 21.72, 4.52, 0.64, 34.56], [0.830, 2.030, 29.35, 4.94, 0.70, 30.41], [0.600, 1.500, 10.34, 3.45, 0.50, 27.47], [0.567, 2.610, 18.41, 4.27, 0.63, 24.04], [
        1.440, -0.750, 24.41, 4.60, 0.72, 20.76], [1.036, 0.710, 23.00, 4.43, 0.74, 18.52], [1.244, -0.840, 27.45, 4.61, 0.76, 16.59], [0.881, 0.453, 25.54, 4.40, 0.79, 14.56], [0.418, 1.950, 28.08, 4.13, 0.79, 13.00]]
    # onny 7 -  15 here...
    zenithLumPerams = zenithLumPeramsDataSet[(skyType-7)]

    return skyPerameters, zenithLumPerams


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


def Lγα(Z, α, sunAlt, sunAzm, skyPerameters):
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


def IrradHarvester(hour, month):

    # Average Hourly Statistics for Global Horizontal Solar Radiation Wh/m≤
    GlobalIrrads = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 35, 40, 23, 3, 0, 0, 0, 0], [0,     0,     12,     59,
                                                              131,    143,    103,     73,     71,     53,      4,      0],
                    [19,       26,     77,    174,    293,    305,
                     253,    224,    203,    169,    109,     47],
                    [92,      136,    215,    338,    473,    482,
                     421,    414,    379,    322,    237,    141],
                    [219,   293,    374,    513,    640,    651,
                     585,    584,    535,    456,    363,    258],
                    [329,   428,    501,    666,    762,    786,
                     751,    739,    668,    558,    457,    356],
                    [410,   508,    599,    778,    826,    879,
                     855,    847,    765,    627,    503,    413],
                    [448,   580,    680,    818,    855,    918,
                     897,    894,    786,    640,    497,    416],
                    [444,   539,    678,    773,    827,    893,
                     880,    873,    743,    585,    442,    369],
                    [387,   472,    605,    667,    729,    810,
                     811,    783,    642,    469,    334,    287],
                    [267,   345,    475,    531,    590,    673,
                     682,    648,    484,    314,    190,    185],
                    [150,   212,    304,    358,    420,    496,
                     511,    464,    295,    151,     68,     78],
                    [45,    84,    134,    176,    239,    304,
                     320,    259,    124,     39,     11,     15],
                    [2,     9,     25,     51,     88,    129,
                     142,     90,     26,      2,      0,      0],
                    [0,     0,      0,      3,     13,     26,
                     28,     11,      1,      0,      0,      0],
                    [0,     0,      0,      0,      0,      0,
                     0,      0,      0,      0,      0,      0],
                    [0,     0,      0,      0,      0,      0,
                     0,      0,      0,      0,      0,      0],
                    [0,     0,      0,      0,      0,      0,
                     0,      0,      0,      0,      0,      0],
                    [0,     0,      0,      0,      0,      0,
                     0,      0,      0,      0,      0,      0],
                    [0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]]

    monthsList = GlobalIrrads[hour]
    globaIrrad = monthsList[month - 1]

    # Average Hourly Statistics for Diffuse Horizontal Solar Radiation Wh/m≤
    #               Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
    DiffuseIrrads = [[0,        0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                     [0,        0,      0,      0,      0,      0,
                         0,      0,      0,      0,      0,      0],
                     [0,     0,      0,      0,      0,      0,
                         0,      0,      0,      0,      0,      0],
                     [0,        0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                     [0,        0,      0,      0,      6,      5,
                      5,      1,      0,      0,      0,      0],
                     [0,      0,      4,     20,     39,     33,
                      34,     21,     21,      7,      0,      0],
                     [6,    11,     31,     67,    104,     87,
                      92,     70,     67,     35,     20,      8],
                     [34,     54,     81,    118,    164,    131,
                      135,    117,    111,     66,     48,     37],
                     [73,    97,    126,    153,    186,    158,
                      158,    146,    138,     91,     72,     76],
                     [101,      136,    155,    176,    185,    168,
                      165,    146,    150,    118,     92,    108],
                     [115,   163,    166,    196,    185,    168,
                      155,    136,    150,    131,    109,    139],
                     [123,   201,    185,    201,    184,    159,
                      140,    128,    143,    128,    113,    152],
                     [127,   196,    183,    188,    181,    149,
                      129,    124,    131,    119,    107,    138],
                     [117,   171,    162,    173,    170,    135,
                      128,    117,    116,    101,     99,    123],
                     [90,   145,    138,    154,    152,    119,
                      117,    111,    101,     79,     81,    100],
                     [65,   104,    105,    124,    132,    103,
                      104,    100,     82,     57,     45,     61],
                     [29,    50,     62,     80,     97,     79,
                      81,     75,     55,     22,     11,     19],
                     [2,      8,     19,     36,     51,     45,
                      50,     39,     22,      2,      0,      0],
                     [0,     0,      0,      4,     11,     10,
                      15,      8,      1,      0,      0,      0],
                     [0,     0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                     [0,     0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                     [0,     0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                     [0,     0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                     [0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]]

    diffuseMonthsList = DiffuseIrrads[hour]
    diffuseIrrad = diffuseMonthsList[month - 1]

    directIrrads = [[0,	    0,	    0,	    0,	    0,	    0,	    0,	    0,	    0,	    0,	    0,	    0	],
                    [	    0,	    0,	    0,	    0,	    0,	    0,
                          0,	    0,	    0,	    0,	    0,	    0	],
                    [0,	    0,	    0,	    0,	    0,	    0,
                     0,	    0,	    0,	    0,	    0,	    0	],
                    [	    0,	    0,	    0,	    0,	    0,	    0,
                          0,	    0,	    0,	    0,	    0,	    0	],
                    [0,	    0,	    0,	    0,	    0,	    1,
                     0,	    0,	    0,	    0,	    0,	    0	],
                    [0,	    0,	    0,	    6,	   27,	   74,
                     50,	   18,	    0,	    0,	    0,	    0	],
                    [	    0,	   17,	   48,	  120	,  161, 240,
                          226	,  212	,  111	,   50	,    2,	    0	],
                    [75,	  207,	  213	,  225,	  285	,  355,
                     345	,  385	,  322	,  336,	  187	,   82	],
                    [289,	  366,	  318,	  349	,  399	,  457	,
                        445, 457, 518, 525	,  451	,  326	],
                    [	  425	,  449,	  380,	  460	,  577,	  565,
                        575,	  585,	  548,	  587,	  555,	  445	],
                    [	  496,	  473	,  446,	  509, 616	,  622,
                        675	,  679,	  670	,  568,	  603	,  495	],
                    [516,	  479,	  490,	  571,	  660,	  673	,
                     729,	  736,	  715	,  636,	  619	,  463	],
                    [503,	  417,	  507,	  603	,  676, 695,
                     745,	  749	,  731,	  673	,  577,	  464	],
                    [	  493,	  412	,  518	,  549,	  654,	  709	,
                        745	,  722,	  731,	  665	,  599,	  376	],
                    [	  429	,  331, 478	,  543,	  603	,  699	,
                        718,	  694,	  721,	  656,	  540	,  360	],
                    [	  341	,  235, 413	,  506, 506	,  638,
                        654	,  614	,  665,	  563,	  416,	  256	],
                    [	  156	,  153,	  279	,  376,	  420	,  552	,
                        565,	  490,	  523	,  394,	  182	,   63	],
                    [0,	    7,	   81	,  215,	  270	,  384,
                     395	,  306,	  195,	   38,	    0,	    0	],
                    [0,	    0,	    0	,   16, 55,	  122,	  111,
                     29	,    0,	    0,	    0	,    0	],
                    [0	,    0	,    0	,    0	,    0,	    0	,
                     0,  0	,    0,	    0	,    0	,    0	],
                    [0,  0	,    0,   0, 0,   0,   0,   0,   0	,    0,   0,	    0	],
                    [0,  0	,    0,   0,  0	,    0,	    0	,
                        0,   0,   0,	    0	,    0	],
                    [0,  0	,    0,   0,   0,   0	,    0,
                        0,   0	,    0	,    0,   0	],
                    [0,   0,   0,  0,   0,   0,   0, 0,   0,   0,   0,   0	]]

    diffuseMonthsList = directIrrads[hour]
    directIrrad = diffuseMonthsList[month - 1]

    return directIrrad, diffuseIrrad, globaIrrad


def calcLumTurbFactor(skyType, sunAlt, month, hour):
    # Caluculate luminous turbidity factor Tv
    # Tv may need to be calculated for each sky type for best reasults.. need function to generalise this...
    # If global illuminance Gv and diffuse illuminance
    # Dv are measured, ...approx or from weather...
    # Evs is the direct-beam illuminance on a horizontal surface (at the ground);
    # Evs = from reaeal sky data..
    sunAlt = math.radians(sunAlt)
    # from weather data...
    # convert from weather file from rad --- illum per thousand...
    # from weather file...vary with time for a particular place...
    # print(IrradHarvester(5,5))
    # check this area
    # globaIrrad, diffuseIrrad = IrradHarvester(hour, month)
    # may be rad and not irrad...
    directIrrad, diffuseIrrad, globaIrrad = IrradHarvester(hour, month)
    irrad2illum = 120
    # conversion correct?...
    globalillum = globaIrrad*irrad2illum
    diffuseillim = diffuseIrrad*irrad2illum
    Evs = (globalillum-diffuseillim)/(1000)
    Evs = directIrrad*irrad2illum/1000
    # needs to be 70...
    # different option use the nomal solar rad values..
    # may need to do an interpelation..
    Ev = 133.8*math.sin(sunAlt)
    # m is the relative optical air mass.
    # check where magic numbers come from.
    m = 1/(math.sin(sunAlt) + 0.50572*(sunAlt + 6.07995)**(-1.6364))
    # av is the luminous extinction coefficient;
    av = 1/(9.9+0.043*m)
    # approximate values for Tv given in "CIE GENERAL SKY STANDARD DEFINING LUMINANCE DISTRIBUTIONS "
    # needs to find evs 27000
    # Evs = 70 maybe will not give output of this sky type...
    Tv = -np.log(Evs/Ev)/(av*m)
    # return Tv
    # actually aiming for 2.5
    TvApprox = [45, 20, 45, 20, 45, 20, 12, 10, 12, 10, 4, 2.5, 4.5, 5, 4]
    Tv = TvApprox[skyType-1]

    # back calc..
    requiredEvs = Ev*np.exp(-3.2*av*m)

    return Tv
    # if no weather data... use approximation..


def compareAndrew():
    # some values outputted from onoine model...
    # model:
    # 1. for clear sky..
    # 2. location lat long : 30, 30
    # 4. timezone 2+
    # 3. date march 21
    # time 10Am
    # diff horizontal 70000
    # position of outcome: azi: eli:

    # {
    #   "azi": 0.6,
    #   "alt": 0.6,
    #   "area": 0.00006980806627785664,
    #   "vector": [ 0.01047121 0.999890342 0.010471784 ],
    #   "luminance": 1.7483077841159684
    # },

    # mine -->8772.68850112396
    # online ...7971.9104316848

    # {
    #     "azi": 57.0,
    #     "alt": 1.8,
    #     "area": 0.00006977744615280901,
    #     "vector": [ 0.838256735 0.544370289 0.031410759 ],
    #     "luminance": 1.6584097194529448
    #   },

    # mine-->9728.21580848599
    # online...7565.312836028867

    # {
    #   "azi": 346.15385,
    #   "alt": 42.6,
    #   "area": 0.00006975791660556778,
    #   "vector": [ -0.176159563 0.714707444 0.67687597 ],
    #   "luminance": 0.4664472823812774
    # },
    # mine -->3082.2922713668695
    # online...2128.4291244683227 relatively far out...

    # {
    #   "azi": 24.08451,
    #   "alt": 61.8,
    #   "area": 0.00006969647315035999,
    #   "vector": [ 0.192840223 0.431412649 0.881303452 ],
    #   "luminance": 0.5673277201978417
    # },
    lum = 0.5673277201978417/(np.pi * 0.00006969647315035999)
    # print(lum)


def getLz(oneSunAlt, zenithLumPerams, skyType, Tv):
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
    # Tv = 3.95
    A = (A1*Tv) + A2
    Lz = A*math.sin(oneSunAlt) + ((0.7*(Tv+1)*(math.sin(oneSunAlt)
                                               ** C))/(((math.cos(oneSunAlt))**D))) + 0.004*Tv
    return Lz*1000


def generateLuminances(sunAlt, sunAzm, zenithLumPerams, skyPerameters, skyType, Tv):
    # Main caluclation
    Lz = getLz(sunAlt, zenithLumPerams, skyType, Tv)
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
    solidAngleMatrix = np.zeros((90, 360))
    for α in range(0, 360):
        for Z in range(0, 90):
            # record value for Z = 0 to use in normalisation
            if Z == 0:
                Lzen = Lγα(Z, α, sunAlt, sunAzm, skyPerameters)
            else:
                # calculate solid angle
                solidAngleMatrix[Z, α] = math.sin(
                    Z*(math.pi/180))*(math.pi/180)*(math.pi/180)

                LγαRatioMatrix[Z, α] = Lγα(Z, α, sunAlt, sunAzm, skyPerameters)

    # need to get solid angle for each degree...
    # multiply ratio by the zenith luminance to get aboslute luminance for every az and el
    toFind = Lγα(90-0.6, 0.6, sunAlt, sunAzm, skyPerameters)*Lz
    horizillum = np.sum(LγαRatioMatrix*Lz*solidAngleMatrix)/(np.pi)

    # print(horizillum)

    lumDist = LγαRatioMatrix*Lz

    return lumDist, toFind, Lz


def generateLum360Images(lumDist):
    import png
    # lumDist = lumDist/600
    # lumDist = np.rint(lumDist)
    # lumDist = lumDist.astype(int)

    # print(lumDist)
    # png.from_array(lumDist, 'L').save("lumDist.png")

    x = np.array(lumDist, np.int32)
    plt.imshow(x)
    plt.savefig("array")
    # plt.imshow(lumDist)
    # plt.show()


def showSky(lumDist):
    # Plot the sky out for user
    a = np.linspace(0, 2*np.pi, 360)
    b = np.linspace(0, 1, 90)
    # actual plotting
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_yticklabels([])
    # Adjust the .001 to get finer gradient
    clev = np.arange(lumDist.min(), lumDist.max(), 1000)
    ctf = ax.contourf(a, b, lumDist, clev, cmap=cm.jet)
    plt.colorbar(ctf)
    plt.show()


def checkingSum(lumDist):

    return diffuseHorizontal


def Main(year, month, day, hour, timezone, lat, lon, skyType):
    # Main function to generate simulation
    skyPerameters, zenithLumPerams = peremeterInitialiser(skyType)
    sunAlt, sunAzm = solarPosition(year, month, day, hour, timezone, lat, lon)

    Tv = calcLumTurbFactor(skyType, sunAlt, month, hour)
    # diffuseHorizontal = checkingSum(lumDist)
    # prin
    # t(Tv)
    if sunAlt > 0:
        lumDist, toFind, Lz = generateLuminances(
            sunAlt, sunAzm, zenithLumPerams, skyPerameters, skyType, Tv)
        print("Lz")
        print(Lz)
        print("specific point")
        print(toFind)
        showSky(lumDist)
        # generateLum360Images(lumDist)
    else:
        print("sundown so cannot display sun distribution")


Main(2020, 3, 21, 10, 3, 54, 44, 12)
# compareAndrew()
