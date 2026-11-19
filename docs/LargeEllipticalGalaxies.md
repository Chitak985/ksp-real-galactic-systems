# Ten more large elliptical galaxies

Added on 2026-11-18 through the existing catalog importer. The saved catalog now
contains 20 generated galaxies. These ten new configs are individual files in
`RealGalacticSystems/Systems`, with no extra galaxy folders.

The selection focuses on large, regular stellar bodies with complete data. It
is not an exhaustive ranking of the universe's ten largest galaxies. Elliptical
and cD galaxies dominate this batch because many extremely extended spiral
examples are distorted mergers, ring systems, or have giant gas disks rather
than equally large stellar disks. Those structures remain outside the supported
templates. A galaxy formed by ancient mergers can still have a regular present
elliptical shape; it is not modeled as an ongoing merger.

## Adopted size and sky geometry

Physical diameters below use the existing cosmology's angular-diameter distance,
not comoving placement distance. Sizes depend on passband and surface-brightness
limit, and are not measurements at a uniform isophote. They omit still fainter
outer envelopes rather than replacing the galaxy with its entire host cluster.
The two ESO objects use optical B-band D26 outlines. The remaining eight use the
saved SIMBAD stellar isophotes. These are conservative render sizes, not the
largest extrapolated diameters sometimes quoted for these galaxies.

| Galaxy | Angular diameter (arcmin) | Stellar diameter (kpc) | Stellar diameter (thousand ly) | Light lookback (Myr) | Sky PA (degrees) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ESO 383-76 | 3.862 | 184.0 | 600 | 545 | 5 |
| ESO 306-17 | 3.250 | 144.1 | 470 | 506 | 178 |
| Holm 15A | 1.558 | 104.4 | 340 | 771 | 148 |
| NGC 6173 | 2.645 | 96.9 | 316 | 417 | 141 |
| NGC 2832 | 3.046 | 87.7 | 286 | 326 | 161 |
| UGC 9799 | 1.987 | 85.1 | 278 | 488 | 34 |
| NGC 1600 | 3.707 | 74.4 | 243 | 227 | 9 |
| NGC 6086 | 1.745 | 70.5 | 230 | 460 | 3 |
| NGC 4874 | 2.340 | 70.3 | 229 | 341 | 63 |
| NGC 1132 | 2.337 | 68.2 | 222 | 331 | 139 |

All ten use the elliptical template. The measured PA and apparent flattening
orient the projected ellipsoid from Sol. The intrinsic three-dimensional shape,
depth, and rotation axis are unknown; the template does not invent a measured
physical disk inclination for an elliptical galaxy. No separate black hole is
created. Each body is one invisible barycenter on a Sun-parented, circular
placement orbit with `period = 3E300` and `mode = OFF`. The rendered orientation
is static and independent of that artificial orbit, with own-axis spin enabled
for any later manual animation. Core SOIs are 100 pc.

## Reviewed measurements and estimates

**ESO 383-76 and ESO 306-17:** saved queries of the
[ESO-LV catalog, VII/115/esolv1](https://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=VII/115/esolv1)
provide D26(B) major diameters of 231.7 and 195.0 arcseconds and adopted major/minor
ratios of 1.99 and 1.52. These replace the smaller catalog summary outlines.
The much larger D90(B) values are retained as provenance only, not rendered as
a sharp stellar edge. PA remains the SIMBAD sky angle.
[ESO catalog documentation](https://archive.eso.org/wdb/help/eso/esolv.html).

**Holm 15A:** stellar mass uses the published estimate of
(2.5 +/- 0.64) trillion solar masses. It is not the separate black-hole mass.
The source warns that extrapolating stellar mass-to-light ratio introduces
uncertainty. The smooth renderer does not reproduce its very depleted core or
a detailed dark halo. [Mehrgan et al. (2019), Section V.1](https://arxiv.org/html/1907.10608).

**NGC 1600:** use B = 11.93 from Table 1 of
[Smith et al. (2008)](https://www.uv.es/ferbaro/papers/ngc1600.pdf), replacing the
SIMBAD summary value B = 8.9. Keeping the discrepant value would overestimate
the displayed surface brightness. Its boxy isophotes remain a smooth ellipsoid
approximation.

**NGC 4874, NGC 6173, and UGC 9799:** detailed stellar-envelope profiles are more
complex than one Gaussian. The renderer retains conservative catalog extents;
it does not add the full intracluster-light component, companion galaxies, or
UGC 9799's active nucleus. [Seigar, Graham & Jerjen (2007)](https://arxiv.org/abs/astro-ph/0612229).

**ESO 306-17 and NGC 1132:** represent their giant elliptical stellar hosts, not
the surrounding fossil group's X-ray halo or dark matter distribution.
[Alamo-Martinez et al. (2012)](https://doi.org/10.1051/0004-6361/201219285).

Except for Holm 15A, gravity uses explicitly estimated stellar masses from Ks
luminosity with mass-to-light ratio 1 solar. These are point-mass game proxies,
not central black-hole masses. Distances use observed redshift as cosmological
redshift; peculiar velocities and the chosen cosmology add uncertainty. The
observed shape is already at its light-emission epoch and is not rewound again.
Each new galaxy's game description briefly states its relevant limitations.

## Rebuild and provenance

Inputs, review notes, and links are in `data/large_elliptical_adoptions.json`.
The ten original SIMBAD/VizieR responses and supplementary ESO-LV responses are
saved under `data/galaxy_sources`. Existing galaxy records and orientation
refinements are preserved. Refreshing needs the reviewed replacement file:

```powershell
python fetch_galaxies.py "ESO 383-76" "ESO 306-17" "Holm 15A" "NGC 6173" "NGC 2832" "UGC 9799" "NGC 1600" "NGC 6086" "NGC 4874" "NGC 1132" --refresh --overrides data/large_elliptical_adoptions.json
python rebuild_galaxies.py --check
```

Add `--offline` to replay the saved responses without network access. Normal
`python rebuild_galaxies.py` uses the saved catalog and templates directly.
Unrelated star and legacy galaxy files, Sun, README, and Git state are preserved.

## Verification

All 41 Python tests pass, including offline source replay and the projection of
all 23 current galaxy disks. The 20 generated galaxy configurations and 150
star configurations match their saved inputs. All ten new galaxies were also
rendered and visually inspected with the actual GalacticDisk renderer and
packaged shaders in an isolated Unity preview, looking from Sol with celestial
north up. The results are visible warm ellipsoids with the adopted sky shapes.
This preview uses KSP API substitutes and no Milky Way foreground or user save;
it is not a live KSP play test or an absolute photometric calibration.
