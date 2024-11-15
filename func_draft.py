from scipy.interpolate import interp1d
from pycbc.catalog import Merger
from pycbc.detector import add_detector_on_earth, Detector
from pycbc.distributions.utils import draw_samples_from_config
from pycbc.filter import matched_filter, sigma, resample_to_delta_t, highpass
from pycbc.psd import from_txt
from pycbc.waveform import get_fd_waveform, get_td_waveform, get_td_waveform_from_fd
from pycbc.waveform.spa_tmplt import findchirp_chirptime
from lunarsky import MoonLocation, MCMF
from astropy import coordinates, constants
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates import Longitude, Latitude, BarycentricTrueEcliptic, CartesianRepresentation, Distance, SkyCoord, get_body_barycentric, AltAz
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pycbc.frame
import pylab
import h5py
import lal

##########################################################################################

class MoonSurfacePoint: #calls position on the lunar surface
    def __init__(self, lunar_det_lat, lunar_det_lon, lunar_det_h):
        self.lat = Latitude(lunar_det_lat * u.rad)
        self.lon = Longitude(lunar_det_lon * u.rad)
        self.h = lunar_det_h * u.m

def gmst_moon(obstime, s_obstime):
    lunar_sidereal_period = 27.321661 * 86400  # Seconds in a sidereal lunar day
    time_utc = Time(obstime, format='gps', scale='utc')
    elapsed_time = time_utc - s_obstime
    gmst_moon = 2 * np.pi * (elapsed_time % lunar_sidereal_period) / lunar_sidereal_period
    return gmst_moon

def set_mspole_location(msp, obstime): #defines detector location in barycentric frame (for LILA case named south pole)
    msp.location = MoonLocation.from_selenodetic(msp.lon, msp.lat, msp.h)
    mcmf_xyz = msp.location.mcmf.cartesian.xyz
    mcmf_skycoord = SkyCoord(mcmf_xyz, frame=MCMF(obstime=obstime))
    barycentric_coords = mcmf_skycoord.transform_to(BarycentricTrueEcliptic(equinox=obstime))
    return barycentric_coords.cartesian.xyz

def obstimes_array(s_obstime, t_obstime, r_obstime):
    num_samples = int(t_obstime / r_obstime) #number of samples
    observation_times = s_obstime + np.arange(num_samples) * r_obstime #sample times array
    return observation_times

def time_delay(observation_times, msp, ra, dec, frame):
    delays = []
    wave_direction = SkyCoord(ra=ra, dec=dec, frame=frame).represent_as(CartesianRepresentation).xyz #represents wave direction
    #print("time_delay: wave direction completed")
    #lists time delays for positions in the sample array
    for obstime in observation_times:
        moon_pos = set_mspole_location(msp, obstime)
        bary_pos = CartesianRepresentation(0, 0, 0)

        moon_proj_distance = np.dot(moon_pos.value.flatten(), wave_direction.value)
        bary_proj_distance = np.dot(bary_pos.xyz.value.flatten(), wave_direction.value)

        distance_difference = (moon_proj_distance - bary_proj_distance) * u.m
        delay = distance_difference / (constants.c.value * u.m / u.s)
        delays.append(delay)
    return delays

def add_detector_on_moon(name, longitude, latitude,
                          yangle=0, xangle=None, height=0,
                          xlength=10000, ylength=10000,xaltitude=0, yaltitude=0):
    _lunar_detectors = {}
    if xangle is None:
            # assume right angle detector if no separate xarm direction given
            xangle = yangle + np.pi / 2.0

        # baseline response of a single arm pointed in the -X direction
    resp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]])
    rm2 = rotation_matrix(-longitude.radian, 'z')
    rm1 = rotation_matrix(-1.0 * (np.pi / 2.0 - latitude.radian), 'y')
        
        # Calculate response in earth centered coordinates
        # by rotation of response in coordinates aligned
        # with the detector arms
    resps = []
    vecs = []
    for angle, azi in [(yangle, yaltitude), (xangle, xaltitude)]:
        rm0 = rotation_matrix(angle * u.rad, 'z')
        rmN = rotation_matrix(-azi *  u.rad, 'y')
        rm = rm2 @ rm1 @ rm0 @ rmN
            # apply rotation
        resps.append(rm @ resp @ rm.T / 2.0)
        vecs.append(rm @ np.array([-1, 0, 0]))

    full_resp = (resps[0] - resps[1])   
    loc = MoonLocation.from_selenodetic(longitude, latitude, height)
    loc = np.array([loc.x.value, loc.y.value, loc.z.value])
    _lunar_detectors[name] = {'location': loc,
                                'response': full_resp,
                                'xresp': resps[1],
                                'yresp': resps[0],
                                'xvec': vecs[1],
                                'yvec': vecs[0],
                                'yangle': yangle,
                                'xangle': xangle,
                                'height': height,
                                'xaltitude': xaltitude,
                                'yaltitude': yaltitude,
                                'ylength': ylength,
                                'xlength': xlength,
                                }

def single_arm_frequency_response(f, n, arm_length): #straight from Detector class
    n = np.clip(n, -0.999, 0.999)
    phase = arm_length / constants.c.value * 2.0j * np.pi * f
    a = 1.0 / 4.0 / phase
    b = (1 - np.exp(-phase * (1 - n))) / (1 - n)
    c = np.exp(-2.0 * phase) * (1 - np.exp(phase * (1 + n))) / (1 + n)
    return a * (b - c) * 2.0 

def antenna_pattern(self, ra, dec, polarization, obstime,
                        frequency=0,
                        polarization_type='tensor'):
        """Return the detector response.

        Parameters
        ----------
        right_ascension: float or numpy.ndarray
            The right ascension of the source
        declination: float or numpy.ndarray
            The declination of the source
        polarization: float or numpy.ndarray
            The polarization angle of the source
        polarization_type: string flag: Tensor, Vector or Scalar
            The gravitational wave polarizations. Default: 'Tensor'

        Returns
        -------
        fplus(default) or fx or fb : float or numpy.ndarray
            The plus or vector-x or breathing polarization factor for this sky location / orientation
        fcross(default) or fy or fl : float or numpy.ndarray
            The cross or vector-y or longitudnal polarization factor for this sky location / orientation
        """
        
        t_gps = obstime.gps
        if isinstance(t_gps, lal.LIGOTimeGPS):
            t_gps = float(t_gps)
        gha = self.gmst_moon(t_gps)

        cosgha = np.cos(gha)
        singha = np.sin(gha)
        cosdec = np.cos(dec)
        sindec = np.sin(dec)
        cospsi = np.cos(polarization)
        sinpsi = np.sin(polarization)

        if frequency:
            e0 = cosdec * cosgha
            e1 = cosdec * -singha
            e2 = np.sin(dec)
            nhat = np.array([e0, e1, e2], dtype=object)

            nx = nhat.dot(self.info['xvec'])
            ny = nhat.dot(self.info['yvec'])

            rx = single_arm_frequency_response(frequency, nx,
                                               self.info['xlength'])
            ry = single_arm_frequency_response(frequency, ny,
                                               self.info['ylength'])
            resp = ry * self.info['yresp'] -  rx * self.info['xresp']
            ttype = np.complex128
        else:
            resp = self.response
            ttype = np.float64

        x0 = -cospsi * singha - sinpsi * cosgha * sindec
        x1 = -cospsi * cosgha + sinpsi * singha * sindec
        x2 =  sinpsi * cosdec

        x = np.array([x0, x1, x2], dtype=object)
        dx = resp.dot(x)

        y0 =  sinpsi * singha - cospsi * cosgha * sindec
        y1 =  sinpsi * cosgha + cospsi * singha * sindec
        y2 =  cospsi * cosdec

        y = np.array([y0, y1, y2], dtype=object)
        dy = resp.dot(y)

        if polarization_type != 'tensor':
            z0 = -cosdec * cosgha
            z1 = cosdec * singha
            z2 = -sindec
            z = np.array([z0, z1, z2], dtype=object)
            dz = resp.dot(z)

        if polarization_type == 'tensor':
            if hasattr(dx, 'shape'):
                fplus = (x * dx - y * dy).sum(axis=0).astype(ttype)
                fcross = (x * dy + y * dx).sum(axis=0).astype(ttype)
            else:
                fplus = (x * dx - y * dy).sum()
                fcross = (x * dy + y * dx).sum()
            return fplus, fcross

        elif polarization_type == 'vector':
            if hasattr(dx, 'shape'):
                fx = (z * dx + x * dz).sum(axis=0).astype(ttype)
                fy = (z * dy + y * dz).sum(axis=0).astype(ttype)
            else:
                fx = (z * dx + x * dz).sum()
                fy = (z * dy + y * dz).sum()

            return fx, fy

        elif polarization_type == 'scalar':
            if hasattr(dx, 'shape'):
                fb = (x * dx + y * dy).sum(axis=0).astype(ttype)
                fl = (z * dz).sum(axis=0)
            else:
                fb = (x * dx + y * dy).sum()
                fl = (z * dz).sum()
            return fb, fl

##########################################################################################

lunar_det_lat = [-(np.pi / 2)]  # Scalar (insert rad)
lunar_det_lon = [0]  # Scalar (insert rad)
lunar_det_h = 0        # Scalar (insert m)
msp = MoonSurfacePoint(lunar_det_lat, lunar_det_lon, lunar_det_h) #create an instance of MoonSurfacePoint

##########################################################################################


add_detector_on_moon("LILA", msp.lon, msp.lat, yangle=0, xlength=10000, ylength=10000)

##########################################################################################

apx = "TaylorF2"
f_lower_LILA = 0.25 #lower bound on frequency sensitivity for LILA and CE
f_final_LILA = 2

t_lower_LILA = 60 * 60 * 24 * 60 #2 months (60 days) pre-merger
t_final_LILA = 60 * 60 * 24 #1 day pre-merger

mass1 = 1.4
mass2 = 1.4
tc = findchirp_chirptime(m1=mass1, m2=mass2, fLower=f_lower_LILA, porder=7)
mass_kg = mass1 * 1.989 * (10**30)

chirp_mass = ((mass_kg*mass_kg)**(3/5))/((mass_kg+mass_kg)**(1/5))
f_lower_t_lower = ((((8*np.pi)**(8/3))/5)*(((constants.G.value*chirp_mass)/(constants.c.value**3))**(5/3))*(tc-t_lower_LILA))**(-3/8)
f_final_t_final = ((((8*np.pi)**(8/3))/5)*(((constants.G.value*chirp_mass)/(constants.c.value**3))**(5/3))*(tc-t_final_LILA))**(-3/8)
#print(f_lower_t_lower, f_final_t_final)

delta_t_LILA = 2 * f_final_t_final
#print(delta_t_LILA)

s_obstime = Time('2024-01-25 01:50:30')  # start time
t_obstime = (t_lower_LILA - t_final_LILA) * u.s # total observation time
r_obstime = delta_t_LILA * u.s  # match observation rate to waveform's delta_t

observation_times = obstimes_array(s_obstime, t_obstime, r_obstime)
print(len(observation_times))
print("obstimes array completed")

ra = 120 * u.deg #right ascension of source
dec = 2 * u.deg #declination of source 
frame = 'icrs' #ref frame for wave direction

delays = time_delay(observation_times, msp, ra, dec, frame)
print("delays array completed")

distance = 1000
inclination = 0.1
duration = tc - t_final_LILA

hp_LILA, hc_LILA = get_td_waveform(approximant=apx, mass1=mass1,
                                            mass2=mass2, f_lower=f_lower_t_lower, f_final = f_final_t_final, delta_t=delta_t_LILA,
                                            inclination=inclination, distance=distance, duration=duration)
print("SSB waveform generation completed")
