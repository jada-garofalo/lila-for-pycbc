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
    lunar_sidereal_period = 27.321661 * 86400  #seconds in a sidereal lunar day
    time_utc = Time(obstime, format='gps', scale='utc')
    elapsed_time = (time_utc - s_obstime).to_value('s')
    gmst_moon = 2 * np.pi * (elapsed_time % lunar_sidereal_period) / lunar_sidereal_period
    return gmst_moon * u.rad

def set_mspole_location_eff(msp, obstimes):
    location = MoonLocation.from_selenodetic(msp.lon, msp.lat, msp.h)
    mcmf_xyz = location.mcmf.cartesian.xyz
    mcmf_skycoords = SkyCoord(mcmf_xyz.T, frame=MCMF(obstime=obstimes))
    barycentric_coords = mcmf_skycoords.transform_to(BarycentricTrueEcliptic(equinox=obstimes))
    xyz_q = barycentric_coords.cartesian.xyz
    return xyz_q.to(u.m)

def time_delay_eff(observation_times, msp, ra, dec, frame='icrs', debug=False): 
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

    k_arr = k_vec.to_value(u.dimensionless_unscaled) #numpy array shape (3,)
    r_arr = moon_positions_m.to_value(u.m) #numpy array shape (3, N)
    proj_m = np.einsum('i...,i...->...', k_arr, r_arr) #projection in meters (numpy array length N)
    proj_m_q = proj_m * u.m

    delays = proj_m_q / constants.c.to(u.m / u.s)

    if debug:
        #print first few values and units for sanity
        print("k (propagation vector, source->SSB):", k_arr)
        print("moon_positions_m[:, :5] (m):", r_arr[:, :5])
        print("proj_m[:5] (m):", proj_m[:5])
    return delays.to(u.s)

def add_detector_on_moon(name, longitude, latitude,
                          yangle=0, xangle=None, height=0,
                          xlength=10000, ylength=10000,xaltitude=0, yaltitude=0):
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

def luna_shift(apx, f_lower_LILA, f_final_LILA, mass1, mass2, delta_t, ra, dec, 
               distance, inclination, start_time, lunar_det_lat=[-(np.pi/2)],
               lunar_det_lon=[0], lunar_det_h=0, frame='icrs', debug=False):

    hp_LILA, hc_LILA = get_td_waveform(approximant=apx, mass1=mass1,
                                                mass2=mass2, f_lower=f_lower_LILA,
                                                f_final = f_final_LILA, delta_t=delta_t,
                                                inclination=inclination, distance=distance)
    
    msp = MoonSurfacePoint(lunar_det_lat, lunar_det_lon, lunar_det_h)
    add_detector_on_moon("LILA", msp.lon, msp.lat)

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
        raise RuntimeError("No overlap after applying delays. "
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

    return hp_LILA_delayed, hc_LILA_delayed, hp_LILA_aligned, hc_LILA_aligned

'''
BELOW ARE SOME TESTS
'''

def ra_dec_for_ecliptic_pole_and_inplane(start_time): 
    if not isinstance(start_time, Time):
        start_time = Time(start_time, scale='utc')

    #ecliptic poles
    pole_n_ecl = SkyCoord(lon=0*u.deg, lat=90*u.deg, frame=BarycentricTrueEcliptic(equinox=start_time))
    pole_s_ecl = SkyCoord(lon=0*u.deg, lat=-90*u.deg, frame=BarycentricTrueEcliptic(equinox=start_time))
    pole_n_icrs = pole_n_ecl.transform_to(ICRS())
    pole_s_icrs = pole_s_ecl.transform_to(ICRS())

    #moon barycentric position at start_time
    moon_bary = get_body_barycentric('moon', start_time)  #CartesianCoord in AU
    moon_icrs = SkyCoord(CartesianRepresentation(moon_bary.x, moon_bary.y, moon_bary.z),
                         frame=ICRS(), representation_type=CartesianRepresentation)
    moon_ecl = moon_icrs.transform_to(BarycentricTrueEcliptic(equinox=start_time))

    #project onto ecliptic plane
    proj_lon = moon_ecl.lon
    inplane_ecl = SkyCoord(lon=proj_lon, lat=0*u.deg, frame=BarycentricTrueEcliptic(equinox=start_time))
    inplane_opposite_ecl = SkyCoord(lon=(proj_lon + 180*u.deg) % (360*u.deg), lat=0*u.deg,
                                    frame=BarycentricTrueEcliptic(equinox=start_time))
    inplane_icrs = inplane_ecl.transform_to(ICRS())
    inplane_opp_icrs = inplane_opposite_ecl.transform_to(ICRS())

    return {
        'pole_n_icrs': pole_n_icrs,
        'pole_s_icrs': pole_s_icrs,
        'inplane_icrs': inplane_icrs,
        'inplane_opp_icrs': inplane_opp_icrs,
        'moon_ecl': moon_ecl,
        'moon_icrs': moon_icrs
    }

start_time = Time('2015-03-17 08:50:00', scale='utc')
dirs = ra_dec_for_ecliptic_pole_and_inplane(start_time)

print("Ecliptic north pole (ICRS): RA = {:.6f} deg, Dec = {:.6f} deg".format(
    dirs['pole_n_icrs'].ra.deg, dirs['pole_n_icrs'].dec.deg))
print("Ecliptic south pole (ICRS): RA = {:.6f} deg, Dec = {:.6f} deg".format(
    dirs['pole_s_icrs'].ra.deg, dirs['pole_s_icrs'].dec.deg))
print("In-plane (aligned to Moon lon) (ICRS): RA = {:.6f} deg, Dec = {:.6f} deg".format(
    dirs['inplane_icrs'].ra.deg, dirs['inplane_icrs'].dec.deg))
print("In-plane opposite (ICRS): RA = {:.6f} deg, Dec = {:.6f} deg".format(
    dirs['inplane_opp_icrs'].ra.deg, dirs['inplane_opp_icrs'].dec.deg))

apx = "TaylorF2"
fhigh = 10
mass1 = 1.4
mass2 = 1.4
delta_t = 1/32.0
distance = 200
inclination = 0.5

dec_vals = np.arange(-80, 80, 5)
ra_vals = np.arange(10, 350, 5)
flows = np.arange(4, 12, 4)

for flow in flows:
    tally = 0
    match_map = np.zeros((len(dec_vals), len(ra_vals)))
    for i, RA in enumerate(ra_vals):
        for j, DEC in enumerate(dec_vals):
            hp_del, hc_del, hp_ali, hc_ali = luna_shift(
                apx=apx, f_lower_LILA=flow, f_final_LILA=fhigh,
                mass1=mass1, mass2=mass2, delta_t=delta_t,
                ra=RA, dec=DEC, distance=distance,
                inclination=inclination, start_time=start_time, debug=False)
            mval = match(hp_ali, hp_del)[0]
            tally = tally + 1
            print(tally, "/2176")
            match_map[j, i] = float(mval)

    RA_grid, DEC_grid = np.meshgrid(ra_vals, dec_vals)
    RA_rad = np.radians(RA_grid)
    DEC_rad = np.radians(DEC_grid)
    RA_rad_shifted = -(RA_rad - np.pi)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="mollweide")
    im = ax.pcolormesh(RA_rad_shifted, DEC_rad, match_map, cmap='viridis', shading='auto')
    ax.grid(True)
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.07, label="Match")
    ax.set_title(f"Match vs Sky Location (flow={flow} Hz)")
    plt.show()

