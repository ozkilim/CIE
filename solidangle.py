
import math
import numpy as np

phispace = np.linspace(0,2*math.pi, 100)

thetaspace = np.linspace(-math.pi/2, math.pi/2, 100)
dtheta = thetaspace[3] - thetaspace[2];
dphi = phispace[3] - phispace[2]


# deltaphispace = np.linspace(0,10,100)
final = 0
for theta in thetaspace:
	for phi in phispace:
		print(dphi*abs((math.cos(theta) - math.cos(theta + dtheta))))
		final = final + dphi*abs((math.cos(theta) - math.cos(theta + dtheta)))


print(final)
