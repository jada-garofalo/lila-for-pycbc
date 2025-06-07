from scipy.interpolate import interp1d, CubicSpline
from pycbc.catalog import Merger
from pycbc.detector import add_detector_on_earth, Detector
from pycbc.distributions.utils import draw_samples_from_config
from pycbc.filter import matched_filter, sigma, resample_to_delta_t, highpass, match
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

'''
PLEASE NOTE:

THIS IS A WORK IN PROGRESS. PLEASE CONTACT JADA GAROFALO AT jagarofa@syr.edu FOR MORE INFORMATION. COMPLETE CLASS IS COMING SOON.

<3 <3 <3
'''

class MoonSurfacePoint: #calls position on the lunar surface
    def __init__(self, lunar_det_lat, lunar_det_lon, lunar_det_h):
        '''
        generates instance of a lunar object to track

        args:
        lunar_det_lat, lunar_det_lon: latitude and longitude in selenodetic coordinates
        expects singleton array with scalar value, assumed in radians
        lunar_det_h: height above the lunar surface, meters, expects ordinary scalar
        '''
        self.lat = Latitude(lunar_det_lat * u.rad)
        self.lon = Longitude(lunar_det_lon * u.rad)
        self.h = lunar_det_h * u.m

def gmst_moon(obstime, s_obstime):
    '''
    generates lunar sidereal time for detector response function

    args:
    obstime: current observation time
    s_obstime: starting observation time

    returns:
    gmst_moon: lunar sidereal time
    '''
    lunar_sidereal_period = 27.321661 * 86400  # Seconds in a sidereal lunar day
    time_utc = Time(obstime, format='gps', scale='utc')
    elapsed_time = time_utc - s_obstime
    gmst_moon = 2 * np.pi * (elapsed_time % lunar_sidereal_period) / lunar_sidereal_period
    return gmst_moon

def set_mspole_location_eff(msp, obstimes):
    location = MoonLocation.from_selenodetic(msp.lon, msp.lat, msp.h)
    mcmf_xyz = location.mcmf.cartesian.xyz
    mcmf_skycoords = SkyCoord(mcmf_xyz.T, frame=MCMF(obstime=obstimes))
    barycentric_coords = mcmf_skycoords.transform_to(BarycentricTrueEcliptic(equinox=obstimes))
    return barycentric_coords.cartesian.xyz  #Return barycentric positions as an array

def obstimes_array(s_obstime, t_obstime, r_obstime):
    num_samples = int(t_obstime / r_obstime) #number of samples
    observation_times = s_obstime + np.arange(num_samples) * r_obstime #sample times array
    return observation_times

def time_delay_eff(observation_times, msp, ra, dec, frame):
    delays = []
    wave_direction = SkyCoord(ra=ra, dec=dec, frame=frame).represent_as(CartesianRepresentation).xyz #represents wave direction
    moon_positions = set_mspole_location_eff(msp, observation_times).value  # shape: (N, 3)
    distance_difference = np.dot(wave_direction, moon_positions) * u.m
    delays = distance_difference / (constants.c.value * u.m / u.s)
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

def luna_shift(apx, f_lower_LILA, f_final_LILA, mass1, mass2, delta_t, ra, dec, 
               distance, inclination, start_time, lunar_det_lat=[-(np.pi/2)],
               lunar_det_lon=[0], lunar_det_h=0, frame='icrs', debug=0):

    '''
    luna_shift generates a BNS waveform in the SSB frame, then generates and applies an
    array of delays to the waveform based on a designated start time to emulate a signal
    response on the lunar surface as it moves

    args:
    apx, f_lower_LILA, f_final_LILA, mass1, mass2, delta_t, ra, dec, distance, inclination:
        all follow standard defs from PyCBC
    frame: designated frame for tracking the moon's spatial evolution, leave as default
        'icrs' for most cases at this point, enter as a string
    lunar_det_lat, lunar_det_lon, lunar_det_h: lunar selenodetic coordinates, defaults to the
        south pole of the moon on the surface. for LILA case leave default. see defaults for
        formatting

    future args:
    detector intrinsic properties such as detector arm orientations and lengths
        (currently left defaulted from prior functions)

    returns:
    hp_LILA, hc_LILA: SSB frame strain
    hp_LILA_delayed, hc_LILA_delayed: lunar frame strain
    delayed_times: array of shifted times (useful for plotting)
    '''

    hp_LILA, hc_LILA = get_td_waveform(approximant=apx, mass1=mass1,
                                                mass2=mass2, f_lower=f_lower_LILA,
                                                f_final = f_final_LILA, delta_t=delta_t,
                                                inclination=inclination, distance=distance)

    msp = MoonSurfacePoint(lunar_det_lat, lunar_det_lon, lunar_det_h)
    add_detector_on_moon("LILA", msp.lon, msp.lat)

    buffer = 540 * u.s #seconds of padding
    s_obstime = start_time - buffer  # start time
    t_obstime = (hp_LILA.duration * u.s) + buffer # total observation time
    r_obstime = 5 * u.s  # match observation rate to waveform's delta_t

    observation_times = obstimes_array(s_obstime, t_obstime, r_obstime)
    delays = time_delay_eff(observation_times, msp, ra, dec, frame)

    hp_LILA_int = CubicSpline(hp_LILA.sample_times, hp_LILA, extrapolate=1)
    hc_LILA_int = CubicSpline(hc_LILA.sample_times, hc_LILA, extrapolate=1)

    observation_times_float = observation_times.gps
    if debug==1:
        print("observation times", observation_times)
        print("observation times gps", observation_times_float)
    delays_int_raw = CubicSpline(observation_times_float, delays, extrapolate=0)
    observation_times_raw = np.linspace(observation_times_float[0], observation_times_float[-1], len(hp_LILA.sample_times))
    delays_int = delays_int_raw(observation_times_raw)
    delayed_times_raw = observation_times_raw - delays_int
    delayed_times = delayed_times_raw - delayed_times_raw[0] + hp_LILA.sample_times[0]

    hp_LILA_delayed = hp_LILA_int(delayed_times)
    hc_LILA_delayed = hc_LILA_int(delayed_times)    
    if debug==1:
        print("original hp_LILA time range:", hp_LILA.sample_times[0], "to", hp_LILA.sample_times[-1])
        print("fixed delayed time range:", delayed_times[0], "to", delayed_times[-1])
        print("hp_LILA_delayed first 10 values:", hp_LILA_delayed[:10])
        print("hp_LILA first 10 values:", hp_LILA[:10])

    return hp_LILA_delayed, hc_LILA_delayed, hp_LILA, hc_LILA, delayed_times

'''
BELOW IS A TEST RUN.
'''

apx = "TaylorF2"
f_lower_LILA = 1
f_final_LILA = 10
mass1 = 1.4
mass2 = 1.4
delta_t = 1/32.0
ra = 20 * u.deg
dec = 10 * u.deg
distance = 200
inclination = 0.5
start_time = Time('2004-06-25 01:50:30')

hp_LILA_delayed, hc_LILA_delayed, hp_LILA, hc_LILA, delayed_times = luna_shift(
    apx=apx, f_lower_LILA=f_lower_LILA, f_final_LILA=f_final_LILA, mass1=mass1,
           mass2=mass2, delta_t=delta_t, ra=ra, dec=dec, distance=distance,
           inclination=inclination, start_time=start_time, debug=1
)

from pycbc.types import TimeSeries

hp_LILA_delayed_ts = TimeSeries(hp_LILA_delayed, delta_t=delta_t, epoch=hp_LILA.start_time)
hc_LILA_delayed_ts = TimeSeries(hc_LILA_delayed, delta_t=delta_t, epoch=hp_LILA.start_time)

t = 0
while t<10:
    t = t + 1
    hp_LILA_delayed_ts = hp_LILA_delayed_ts[:-1]
    hp_LILA = hp_LILA[:-1]
    print(hp_LILA_delayed, hp_LILA_delayed_ts)
    print(hp_LILA, hp_LILA_delayed_ts)
    m = match(hp_LILA, hp_LILA_delayed_ts, low_frequency_cutoff=1.0)
    print("Match", m)
    if t>10:
        break
