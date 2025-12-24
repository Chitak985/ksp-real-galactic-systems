# AstroToKSP.py
# Copyright (C) 2025 Chitak985YT
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

import numpy as np
from astropy import units as u
from astropy.constants import G, M_sun
from astropy.coordinates import Galactocentric, SkyCoord
from astroquery.simbad import Simbad

def getSimbadStar(object):
    custom_simbad = Simbad()

    custom_simbad.add_votable_fields(
        'ra(d)',
        'dec(d)',
        'pmra',
        'pmdec',
        'plx',
        'rv_value'
    )
    
    result = custom_simbad.query_object("Tau Ceti")
    
    ra = float(result['RA_d'][0])
    dec = float(result['DEC_d'][0])
    pmra = float(result['PMRA'][0])
    pmdec = float(result['PMDEC'][0])
    distance = 1000.0 / float(result['PLX_VALUE'][0])
    rv = float(result['RV_VALUE'][0])
    
    return SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=distance * u.pc,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=rv * u.km / u.s,
        frame="icrs"
    )

# ---- CONFIG ----
M_ENC = 1.1e11 * M_sun  # enclosed galactic mass
MU = (G * M_ENC).to(u.m**3 / u.s**2).value

# ---- MAIN FUNCTION ----
def skycoord_to_ksp_orbit(sc: object):
    # Convert to Galactocentric frame
    gal = sc.transform_to(Galactocentric())

    r = gal.cartesian.xyz.to(u.m).value
    v = gal.velocity.d_xyz.to(u.m / u.s).value

    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)

    h = np.cross(r, v)
    hmag = np.linalg.norm(h)

    n = np.cross([0, 0, 1], h)
    nmag = np.linalg.norm(n)

    evec = (np.cross(v, h) / MU) - (r / rmag)
    e = np.linalg.norm(evec)

    energy = vmag**2 / 2 - MU / rmag
    a = -MU / (2 * energy)

    i = np.arccos(h[2] / hmag)

    Omega = np.arccos(n[0] / nmag) if nmag > 0 else 0.0
    if n[1] < 0:
        Omega = 2 * np.pi - Omega

    omega = np.arccos(np.dot(n, evec) / (nmag * e)) if nmag > 0 else 0.0
    if evec[2] < 0:
        omega = 2 * np.pi - omega

    # ---- FIXES FOR KSP ----

    # Convert retrograde to prograde equivalent
    if i > np.pi / 2:
        i = np.pi - i
        Omega = (Omega + np.pi) % (2 * np.pi)
        omega = (omega + np.pi) % (2 * np.pi)

    # Mean anomaly forced to 0 for stability
    M0 = 0.0

    return {
        "semiMajorAxis": a,
        "eccentricity": e,
        "inclination": np.degrees(i),
        "longitudeOfAscendingNode": np.degrees(Omega),
        "argumentOfPeriapsis": np.degrees(omega),
        "meanAnomalyAtEpoch": M0
    }

# ---- Old table data retrieval ----
'''
# ---- DATA RETRIEVAL ----
#tmp = input("Paste table from cdsportal.u-strasbg.fr containing the needed object here and press Enter: ")
tmp = """0.0	* tau Cet e	Planet	01 44 04.08314	-15 56 14.9276	0.119	0.083	90	-1721.728	854.963											ref (26)	26.017013	-15.93748
0.0	* tau Cet b	Planet?	01 44 04.08314	-15 56 14.9276	0.119	0.083	90	-1721.728	854.963											ref (7)	26.017013	-15.93748
0.0	* tau Cet c	Planet?	01 44 04.08314	-15 56 14.9276	0.119	0.083	90	-1721.728	854.963											ref (6)	26.017013	-15.93748
0.0	* tau Cet d	Planet?	01 44 04.08314	-15 56 14.9276	0.119	0.083	90	-1721.728	854.963											ref (6)	26.017013	-15.93748
0.0	* tau Cet f	Planet	01 44 04.08314	-15 56 14.9276	0.119	0.083	90	-1721.728	854.963											ref (19)	26.017013	-15.93748
0.0	* tau Cet g	Planet	01 44 04.08314	-15 56 14.9276	0.119	0.083	90	-1721.728	854.963											ref (16)	26.017013	-15.93748
0.0	* tau Cet h	Planet	01 44 04.08314	-15 56 14.9276	0.119	0.083	90	-1721.728	854.963											ref (13)	26.017013	-15.93748
0.0	* tau Cet	PM*	01 44 04.08314	-15 56 14.9276	0.119	0.083	90	-1721.728	854.963	4.22	3.5	2.88	2.14	1.72	1.68	G8V				ref (1320)	26.017013	-15.93748
91.6	Gaia DR3 2452379051312384000	Star	01 43 59.15172	-15 55 17.1937	0.102	0.071	90	2.103	-0.621											ref (1)	25.996466	-15.921443
100.6	JCMTSF J014402.8-155436	Radio(sub-mm)	01 44 02.8	-15 54 36	10000	10000	90													ref (1)	26.011667	-15.91
109.3	JCMTSF J014406.2-155430	Radio(sub-mm)	01 44 06.2	-15 54 30	10000	10000	90													ref (1)	26.025833	-15.908333
136.1	UCAC4 371-001887	Candidate_Hsd	01 44 07.71087	-15 58 20.6167	0.025	0.021	90	-2.266	-12.868	14.055	13.952	14.105	13.732	13.667	13.689					ref (1)	26.032129	-15.972394
220.1	Gaia DR3 2452378604635585280	Star	01 43 48.86931	-15 56 32.1434	0.055	0.04	90	-0.151	-3.543											ref (1)	25.953622	-15.942262
259.3	MCG-03-05-018	EmG	01 43 46.87	-15 57 29.7						14		14.57					1.18772	0.868577	168	ref (11)	25.945292	-15.95825
325.9	[BA2007] 20	QSO_Candidate	01 44 24.457	-15 58 35.94																ref (1)	26.101904	-15.97665
362.3	[BA2007] 24	QSO_Candidate	01 44 24.443	-15 59 47.22																ref (1)	26.101846	-15.99645
506.5	2MASX J01442194-1548597	Galaxy	01 44 21.96099	-15 48 59.0081	4.041	3.933	90						15.002	14.362	13.702		0.137	0.137	90	ref (1)	26.091504	-15.816391
526.9	LEDA 902795	Galaxy	01 44 17.5	-16 04 25													0.31	0.23	26	ref (1)	26.072917	-16.073611
677.1	BD-16 297	Star	01 44 39.18302	-16 03 44.7790	0.01	0.009	90	3.33	-8.799	11.77	10.96		8.983	8.392	8.224					ref (2)	26.163263	-16.062439
705.8	LEDA 902488	Galaxy	01 43 36.0	-16 05 53	"""
distance = float(input("Enter distance in parsec (pc) units: (can be found on Wikipedia) "))
radialVelocity = float(input("Enter radial velocity in km/s here: (can be found on Wikipedia) "))

# ---- DATA FORMATTING ----
tmp = tmp.split("\n")                                               # Separate into a list of lines
tmp2 = []                                                           # Create a list for formatted rows
for i in tmp:                                                       # Convert each line into a formatted row
    tmp2.append(i.split("\t"))                                      # Convert by splitting the line using \t
tmp3 = ""                                                           # Create a string for a user-friendly list
for i in range(len(tmp2)):                                          # Iterate through every formatted table row
    tmp3 += str(i + 1) + ". " + tmp2[i][1] + "\n"                   # Create the user-friendly list with format "[number]. [object]\n"
print(tmp3)                                                         # Print the user-friendly object list
tmp3 = int(input("Enter the number of the needed object:"))         # Prompt the user to select an object
tmp = tmp2[tmp3 - 1]                                                # Reuse the line list variable for the object row
del tmp3                                                            # Delete the no longer needed user-friendly list
tmp2 = []                                                           # Reuse the formatted rows list variable for a formatted object
for i in tmp:                                                       # Iterate and convert the object
    try: tmp2.append(float(i))                                      # Attempt to convert item to a float
    except Exception: tmp2.append(i)                                # If conversion fails, keep as is
del tmp                                                             # Delete the no longer needed object row variable

# ---- DATA PREPARATION ----
# Links
# Portal:    https://cdsportal.u-strasbg.fr/?target=tau%20ceti
# Wikipedia: https://en.wikipedia.org/wiki/Tau_Ceti
tmp = SkyCoord(
    ra=tmp2[-2] * u.deg,                          # 26.0170   Portal:     RAdeg (deg)
    dec=tmp2[-1] * u.deg,                         # -15.9375  Portal:     DEdeg (deg)
    distance=distance * u.pc,                     # 3.65      Wikipedia:  Distance (the one that uses the parsec/pc units)
    pm_ra_cosdec=tmp2[8] * u.mas / u.yr,          # -1721.0   Portal:     PMRA (mas.yr-1)
    pm_dec=tmp2[9] * u.mas / u.yr,                # 854.0     Portal:     PMDEC (mas.yr-1)
    radial_velocity=radialVelocity * u.km / u.s,  # -16.7     Wikipedia:  Radial Velocity (Rv)
    frame="icrs"                                  # "icrs"    ?????:      ?????
)
'''

# ---- DATA CONVERSION ----
def makeKoprnicusOrbit(orbit):
    return f"""Orbit
    {{
        referenceBody = SagittariusA
        semiMajorAxis = {orbit["semiMajorAxis"]}
        eccentricity = {orbit["eccentricity"]}
        inclination = {orbit["inclination"]}
        longitudeOfAscendingNode = {orbit["longitudeOfAscendingNode"]}
        argumentOfPeriapsis = {orbit["argumentOfPeriapsis"]}
        meanAnomalyAtEpoch = {orbit["meanAnomalyAtEpoch"]}
        epoch = 0
    }}"""