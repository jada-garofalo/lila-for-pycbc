import luna_push as lp
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from pycbc.filter import match

lp = lp.luna_push()

start_time = Time('2015-03-17 08:50:00', scale='utc')
fhigh = 10
mass1 = 1.4
mass2 = 1.4
delta_t = 1/32.0
distance = 200
inclination = 0.5

dec_vals = np.arange(-80, 80, 5)
ra_vals = np.arange(10, 350, 5)
flows = np.arange(8, 12, 4)

for flow in flows:
    tally = 0
    match_map = np.zeros((len(dec_vals), len(ra_vals)))
    for i, RA in enumerate(ra_vals):
        for j, DEC in enumerate(dec_vals):
            print(f"init loop {flow} {i} {j}")
            hp_del, hc_del, hp_ali, hc_ali, det = lp.luna_shift(
                f_lower_LILA=flow, f_final_LILA=fhigh,
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

