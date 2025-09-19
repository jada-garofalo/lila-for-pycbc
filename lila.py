import numpy as np
import matplotlib.pyplot as plt
import pylab
import h5py
import lal

import astropy.units as u
from astropy import coordinates, constants
from astropy.coordinates import (
    Longitude, Latitude, BarycentricTrueEcliptic, CartesianRepresentation,
    Distance, SkyCoord, get_body_barycentric, AltAz, ICRS)
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.time import Time

import pycbc.frame
from pycbc.catalog import Merger
from pycbc.detector import add_detector_on_earth, Detector
from pycbc.distributions.utils import draw_samples_from_config
from pycbc.filter import matched_filter, sigma, resample_to_delta_t, highpass, match
from pycbc.psd import from_txt
from pycbc.types import TimeSeries
from pycbc.waveform import get_fd_waveform, get_td_waveform, get_td_waveform_from_fd
from pycbc.waveform.spa_tmplt import findchirp_chirptime

from scipy.interpolate import interp1d, CubicSpline
from lunarsky import MoonLocation, MCMF

class MoonSurfacePoint:
    '''
    Creates a point on the lunar surface to track
    '''
    def __init__(self, lunar_det_lat, lunar_det_lon, lunar_det_h):
        self.lat = Latitude(lunar_det_lat * u.rad)
        self.lon = Longitude(lunar_det_lon * u.rad)
        self.h = lunar_det_h * u.m

def gmst_moon(obstime, s_obstime):
    '''
    Returns GMST rotation evolution in radians for the moon

    Params
    ------
    obstime: current observation time
    s_obstime: starting observation time

    Returns
    ------
    gmst_moon * u.rad: GMST angle evolved since start time, in radians
    '''
    lunar_sidereal_period = 27.321661 * 86400
    time_utc = Time(obstime, format='gps', scale='utc')
    elapsed_time = (time_utc - s_obstime).to_value('s')
    gmst_moon = 2 * np.pi * (elapsed_time % lunar_sidereal_period) / lunar_sidereal_period
    return gmst_moon * u.rad

def set_mspole_location_eff(msp, obstimes):
    '''
    Returns MoonSurfacePoint location in solar system barycentric coordinates over a set
    collection of observation times

    Params
    ------
    msp: MoonSurfacePoint to track
    obstimes: Array of observation times

    Returns
    ------
    xyz_q.to(u.m): Array of barycentric coordinate positions (in meters) of the
    Moon for given observation times
    '''
    location = MoonLocation.from_selenodetic(msp.lon, msp.lat, msp.h)
    mcmf_xyz = location.mcmf.cartesian.xyz
    mcmf_skycoords = SkyCoord(mcmf_xyz.T, frame=MCMF(obstime=obstimes))
    barycentric_coords = mcmf_skycoords.transform_to(BarycentricTrueEcliptic(equinox=obstimes))
    xyz_q = barycentric_coords.cartesian.xyz
    return xyz_q.to(u.m)

def time_delay_eff(observation_times, msp, ra, dec, frame='icrs', debug=False): 
    '''
    Returns array of time delays for amount of time it takes light to travel from the solar system barycenter
    to the chosen MoonSurfacePoint across an array of observation times

    Params
    ------
    observation_times: Array of observation times to compute delays for
    msp: MoonSurfacePoint to track
    ra: Right ascension of GW source
    dec: Declination of GW source
    frame: Reference coordinate frame, defaults to ICRS
    debug: Verbose debug prints, defaults to False

    Returns
    ------
    delays.to(u.s): Array of light delay times found for each observation time
    '''
    if not isinstance(ra, u.Quantity):
        ra = ra * u.deg
    if not isinstance(dec, u.Quantity):
        dec = dec * u.deg
    
    src = SkyCoord(ra=ra, dec=dec, frame=ICRS()) if (isinstance(frame, str) and frame.lower()=='icrs') \
          else SkyCoord(ra=ra, dec=dec, frame=frame)
    src_ecl = src.transform_to(BarycentricTrueEcliptic(equinox=observation_times))
    src_cart_rep = src_ecl.represent_as(CartesianRepresentation)
    src_vec = src_cart_rep.xyz / src_cart_rep.norm()
    k_vec = (-1.0) * src_vec

    moon_positions_q = set_mspole_location_eff(msp, observation_times)
    if not isinstance(moon_positions_q, u.Quantity):
        raise ValueError("set_mspole_location_eff must return an astropy Quantity with distance units.")
    moon_positions_m = moon_positions_q.to(u.m)
    if moon_positions_m.shape[0] != 3 and moon_positions_m.shape[-1] == 3:
        moon_positions_m = moon_positions_m.T

    k_arr = k_vec.to_value(u.dimensionless_unscaled)
    r_arr = moon_positions_m.to_value(u.m)
    proj_m = np.einsum('i...,i...->...', k_arr, r_arr)
    proj_m_q = proj_m * u.m

    delays = proj_m_q / constants.c.to(u.m / u.s)

    if debug:
        print("k (propagation vector, source->SSB):", k_arr)
        print("moon_positions_m[:, :5] (m):", r_arr[:, :5])
        print("proj_m[:5] (m):", proj_m[:5])
    return delays.to(u.s)

def add_detector_on_moon(name, longitude, latitude,
                          yangle=0, xangle=None, height=0,
                          xlength=10000, ylength=10000,xaltitude=0, yaltitude=0):
    '''
    Constructs a two-arm GW detector for an identified location on the lunar surface

    Params
    ------
    name: Chosen detector name
    longitude: Longitude of detector, can be taken from MoonSurfacePoint attribute
    latitude: Latitude of detector, can be taken from MoonSurfacePoint attribute
    yangle: Angle at which the y-arm of the detector rests, defaults to 0
    xangle: Angle at which the x-arm of the detector rests, defaults to 90 degrees past yangle (right angle detector)
    xlength: Length of x-arm, defaults to 10000m
    ylength: Length of y-aarm, defaults to 10000m
    xaltitude: Altitude of x-arm, defaults to 0m
    yaltitude: Altitude of y-arm, defaults to 0m

    Returns
    ------
    _lunar_detectors: Dictionary containing defined detector
    '''
    _lunar_detectors = {}
    if xangle is None:
            #assume right angle detector if no separate xarm direction given
            xangle = yangle + np.pi / 2.0

        #baseline response of a single arm pointed in the -X direction
    resp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]])
    rm2 = rotation_matrix(-longitude.radian, 'z')
    rm1 = rotation_matrix(-1.0 * (np.pi / 2.0 - latitude.radian), 'y')
        
        #Calculate response in earth centered coordinates
        #by rotation of response in coordinates aligned
        #with the detector arms
    resps = []
    vecs = []
    for angle, azi in [(yangle, yaltitude), (xangle, xaltitude)]:
        rm0 = rotation_matrix(angle * u.rad, 'z')
        rmN = rotation_matrix(-azi *  u.rad, 'y')
        rm = rm2 @ rm1 @ rm0 @ rmN
            #apply rotation
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
    return _lunar_detectors

class LunarDetector:
    '''
    Wrapper for detector dict information and squeezed response
    '''
    def __init__(self, detdict, s_obstime):
        self.info = detdict
        self.response = np.squeeze(detdict['response'])
        self.s_obstime = s_obstime
    def gmst_moon(self, obstime):
        return gmst_moon(obstime, self.s_obstime)

def luna_shift(apx, f_lower_LILA, f_final_LILA, mass1, mass2, delta_t, ra, dec, 
               distance, inclination, start_time, lunar_det_lat=[-(np.pi/2)],
               lunar_det_lon=[0], lunar_det_h=0, frame='icrs', debug=False):
    '''
    Generates a waveform as defined by the user, and then applies a time delay shift to said waveform based on
    a given observation period and lunar detector information

    Params
    ------
    apx: Waveform approximant
    f_lower_LILA: Lower frequency bound for waveform generation
    f_final_LILA: Upper frequency bound for waveform generation
    mass1: Mass of first body for waveform generation
    mass2: Mass of second body for waveform generation
    delta_t: Sample rate for time domain waveform
    ra: Right ascension of GW source
    dec: Declination of GW source
    distance: Distance of source (Mpc)
    inclination: Inclination of source
    start_time: Starting time of observation period
    lunar_det_lat: Latitude of lunar detector, defaults to South pole
    lunar_det_lon: Longitude of lunar detector, defaults to South pole
    frame: Reference coordinate frame, defaults to ICRS
    debug: Verbose debug prints, defaults to False

    Returns
    ------

    hp_LILA_delayed: Delayed h_plus time series
    hc_LILA_delayed: Delayed h_cross time series
    hp_LILA_aligned: Undelayed h_plus time series
    hc_LILA_aligned: Undelayed h_cross time series
    _lunar_detectors: Lunar detector dictionary (needed for antenna response calculation)

    Note: named after Jada's cat, Luna >^^<
    '''

    hp_LILA, hc_LILA = get_td_waveform(approximant=apx, mass1=mass1,
                                                mass2=mass2, f_lower=f_lower_LILA,
                                                f_final = f_final_LILA, delta_t=delta_t,
                                                inclination=inclination, distance=distance)
    
    msp = MoonSurfacePoint(lunar_det_lat, lunar_det_lon, lunar_det_h)
    _lunar_detectors = add_detector_on_moon("LILA", msp.lon, msp.lat)

    t_det = np.asarray(hp_LILA.sample_times)
    t0 = t_det[0]
    t_abs = start_time + (t_det - t0) * u.s
    delays = time_delay_eff(t_abs, msp, ra, dec, frame=frame, debug=debug)
    delays_s = np.asarray(delays.to_value(u.s))
    
    if debug:
        print("delays[:5] (s):", delays_s[:5])
        print("delays[-5:] (s):", delays_s[-5:])

    t_ssb = t_det + delays_s
    t_min = t_det[0]
    t_max = t_det[-1]

    valid = (t_ssb >= t_min) & (t_ssb <= t_max)
    if not np.any(valid):
        raise RuntimeError("No overlap after applying delays."
                           "Increase waveform duration or choose a different start_time.")
    
    hp_int = CubicSpline(t_det, np.asarray(hp_LILA), extrapolate=False)
    hc_int = CubicSpline(t_det, np.asarray(hc_LILA), extrapolate=False)

    hp_vals_delayed = hp_int(t_ssb[valid])
    hc_vals_delayed = hc_int(t_ssb[valid])

    hp_vals_orig = np.asarray(hp_LILA)[valid]
    hc_vals_orig = np.asarray(hc_LILA)[valid]

    epoch_shift = int(valid.argmax()) * delta_t
    epoch = hp_LILA.start_time + epoch_shift 

    hp_LILA_delayed = TimeSeries(hp_vals_delayed, delta_t=delta_t, epoch=epoch)
    hc_LILA_delayed = TimeSeries(hc_vals_delayed, delta_t=delta_t, epoch=epoch)
    hp_LILA_aligned = TimeSeries(hp_vals_orig, delta_t=delta_t, epoch=epoch)
    hc_LILA_aligned = TimeSeries(hc_vals_orig, delta_t=delta_t, epoch=epoch)

    if debug:
        print(f"Original t range: [{t_min:.6f}, {t_max:.6f}] s")
        print(f"Delayed t range (kept): [{t_ssb[valid][0]:.6f}, {t_ssb[valid][-1]:.6f}] s")
        print(f"Overlap points: {np.count_nonzero(valid)}/{len(valid)}")

    return hp_LILA_delayed, hc_LILA_delayed, hp_LILA_aligned, hc_LILA_aligned, _lunar_detectors

def single_arm_frequency_response_alt(f, n, arm_length):
    '''
    Same as in pycbc.detector
    '''
    n = np.clip(n, -0.999, 0.999)
    phase = arm_length / constants.c.value * 2.0j * np.pi * f
    a = 1.0 / 4.0 / phase
    b = (1 - np.exp(-phase * (1 - n))) / (1 - n)
    c = np.exp(-2.0 * phase) * (1 - np.exp(phase * (1 + n))) / (1 + n)
    return a * (b - c) * 2.0

def antenna_pattern_alt(self, ra, dec, polarization, obstime,
                        frequency=0,
                        polarization_type='tensor'):
    '''
    Same as in pycbc.detector, except with added obstime (observation time) param
    to find orientation angle as the Moon rotates over the observation period
    '''
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

            rx = single_arm_frequency_response_alt(frequency, nx,
                                               self.info['xlength'])
            ry = single_arm_frequency_response_alt(frequency, ny,
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

def time_dependent_strain(detector, hp, hc, ra, dec, polarization, debug=False):
    '''
    Returns strain for given delayed waveform and detector configuration
    Intended use is after a run of luna_shift()
    '''
    fp_t = []
    fc_t = []
    for t in hp.sample_times:
        obstime = Time(float(hp.start_time + t), format='gps', scale='utc')
        fp_i, fc_i = antenna_pattern_alt(detector,
                                     ra=ra,
                                     dec=dec,
                                     polarization=polarization,
                                     obstime=obstime)
        fp_t.append(fp_i)
        fc_t.append(fc_i)
        if debug:
            print("tick",t)
    fp_t = np.array(fp_t)
    fc_t = np.array(fc_t)
    return fp_t * hp + fc_t * hc

