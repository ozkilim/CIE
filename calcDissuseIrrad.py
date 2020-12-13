# Solar constant W/m^2 
DC_SolarConstantE = 1367.0; 

double CalcEccentricity()

	double day_angle;	/* Day angle (radians) */
	double E0;			/* Eccentricity */

	/* Calculate day angle */
	day_angle  = (julian_date - 1.0) * (2.0 * PI / 365.0);

	/* Calculate eccentricity */
	E0 = 1.00011 + 0.034221 * cos(day_angle) + 0.00128 * sin(day_angle)
			+ 0.000719 * cos(2.0 * day_angle) + 0.000077 * sin(2.0 *
			day_angle);

	return E0;

double CalcAirMass()

	return (1.0 / (cos(sun_zenith) + 0.15 * pow(93.885 -
			RadToDeg(sun_zenith), -1.253)));


double CalcSkyBrightness()

	return diff_irrad * CalcAirMass() / (DC_SolarConstantE *
			CalcEccentricity());


sky_brightness = CalcSkyBrightness()

CalcDiffuseIrradiance = (sky_brightness * DC_SolarConstantE * CalcEccentricity()) / CalcAirMass();






			