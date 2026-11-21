# GN-z11 and JADES-GS-z14-0 host visuals

Both existing manual bodies now have a `GalacticDiskDefinition`. Each uses a smooth, compact envelope for the resolved young stellar population. No mature spiral arms, dust lanes, bar, or individually resolved emission knots are inferred. GN-z11 retains its separate existing central accretion effect. JADES retains its invisible barycenter.

The descriptions in the game mention the visual uncertainties. These objects are observed at their emission epochs; the visuals do not extrapolate their evolution to the present day.

## Adopted imaging measurements

| Body | Angular major-axis half-light radius | Apparent minor/major ratio | Adopted major-axis PA, east of north |
| --- | --- | --- | --- |
| GN-z11 resolved host | 0.049 arcsec | 0.67 | 34 degrees |
| JADES-GS-z14-0 | 0.0788 arcsec | 0.425 | 8.2389 degrees |

[Tacchella et al. (2023), section 3.2 and Table 2](https://arxiv.org/html/2302.07234v2) measure GN-z11's extended component at 49 +/- 3 milliarcsec (about 200 pc), axis ratio 0.67 +/- 0.05, Sersic index 0.9 +/- 0.1 and PA 34 +/- 5 degrees. The much smaller combined 64 +/- 20 pc size includes the nuclear point source; it is not the host-envelope size. The 21-milliarcsec nucleus/host centroid offset and the uncertain northeastern haze are omitted. The published PA is adopted as a conventional sky major-axis angle; fitting-method conventions and the unknown three-dimensional geometry add uncertainty.

[Helton et al. (2025), section 1.3.1](https://arxiv.org/html/2405.18462v3) give JADES's radius as 0.0788 +/- 0.0006 arcsec, axis ratio 0.425 +/- 0.008, Sersic index 0.877 +/- 0.027 and ForcePho angle 1.427 +/- 0.008 radians. [ForcePho's source definition](https://raw.githubusercontent.com/bd-j/forcepho/main/forcepho/sources.py) specifies north-of-east angles, consistent with its [rotation matrix in east/north coordinates](https://raw.githubusercontent.com/bd-j/forcepho/main/forcepho/slow/gaussmodel.py). The adapter uses east of north, so this raw angle becomes `90 - degrees(1.427)`, modulo 180 degrees. This is the observed, potentially lensed image-plane shape; no second lensing correction is applied.

The full measurements, source links, assumptions and display scales are saved in [the legacy orientation registry](../data/legacy_disk_orientations.json). Existing positions and redshifts remain sourced from [the manual position registry](../data/manual_object_positions.json).

## Size and shape conversion

The existing frozen Sun orbits place GN-z11 and JADES at approximately 9,755 and 10,338 Mpc in comoving coordinates. Their angular-diameter distances are approximately 841 and 681 Mpc. Multiplying a proper emission-epoch size by `D_comoving / D_angular` preserves its observed angle in this static scene. Using proper radii directly at the comoving positions would make them too small.

The renderer's smooth Gaussian kernel is used as an approximation to each measured near-exponential light profile. With normalized `galaxyRadius = 100`, `bulgeScaleX = 100/3.5`, and scaled-space factor 6000:

```text
angular_sigma = radians(half_light_arcsec / 3600) / sqrt(2 * ln(2))
objectScale = D_comoving_m * tan(angular_sigma) / (6000 * bulgeScaleX)
bulgeScaleZ = axis_ratio * bulgeScaleX
bulgeScaleY = sqrt(axis_ratio) * bulgeScaleX
```

For an optically thin projected Gaussian, the half-light ellipse lies at `sqrt(2 ln 2)` sigma. This preserves the measured geometric half-light scale, corresponding to about 200 and 260 proper pc with the saved cosmology. Tone mapping and the renderer's internal attenuation are not a fit to the observed surface-brightness profiles. The Gaussian falls off faster than a true exponential at large radii.

The local Y axis faces the observer and X follows the adopted sky major axis, using the same ICRS-to-Galactic-to-Unity adapter as other galaxies. The fitted elongation is supplied directly by the X/Z scales. `inclination_deg = 0` in these two registry records describes that modeling convention, not a measured face-on physical disk. Line-of-sight depth is an estimate; no spin axis, near side or dynamical rotation period is claimed. `alignRotationToOrbit = false` and `rotationPeriod = 0` keep the visuals fixed even though GN-z11's central body rotates.

## Display choices and maintenance

The detected rest-UV light arrives in the infrared. Pale blue maps that light into visible display colors, allowing these hosts to be inspected in the game; it is not their naked-eye color. Exposure, star particle brightness, depth and fine structure are estimates. The 2,000 points per host are display samples, not a star count. GN-z11's existing accretion effect is not photometrically matched to its host. At their true apparent sizes, both galaxies are unresolved at ordinary field of view; zooming or approaching is needed to inspect the envelope.

Both definitions use the existing opacity presets: inside 0.2, daylight atmosphere 0.01, night atmosphere 0.65. Those are gameplay visibility settings.

The definitions are appended to the two existing manual configuration files. `rebuild_manual_orbits.py` preserves them, and `rebuild_galaxies.py` continues to exclude these manual bodies. If changing their positions, recompute the display scale from the angular radius above and update the registry alongside the configuration. Their root, frozen orbits, compact SOIs, masses, lighting and central effects are otherwise unchanged.
