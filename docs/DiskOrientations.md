# Galaxy disk orientations

All 2,060 current `GalacticDiskDefinition` nodes have saved orientation inputs.
The original 13 were reviewed on 2026-11-18; ten additional large ellipticals
use catalog position angles and apparent flattening from their saved records.
[Thirty more spirals](SpiralGalaxies.md) add published disk inclinations for 28
objects and explicit axial-ratio estimates for two; their full orientation table
and source uncertainties are documented separately.
[Fifty nearby dwarf spheroidals](NearbyGalaxies.md) add measured projected
major-axis position angles and axial ratios; Crater II explicitly uses a
near-circular upper-limit approximation. These shapes do not establish physical
spin axes or disk inclinations.
[Nine hundred further galaxies](BulkGalaxies.md) retain source position angles
and flattening; the new sample spans all nine supported regular families.
[A further thousand galaxies](SecondThousandGalaxies.md) add source-based
orientations across six regular galaxy families.
[Fifteen reviewed nearby and forty-two distant galaxies](FarAndNearbyGalaxies.md)
add measured sky geometry and explicit research assumptions.
These settings describe their observed orientation, not a rigid rotation speed.
The observed light already represents the emission epoch. Disks remain static
with `rotationPeriod = 0` and `alignRotationToOrbit = false`. They also explicitly
set `spinAroundDiskAxis = true`, so enabling spin later will not precess a tilted
disk around the world's vertical axis.

Position angle (PA) is east of celestial north, modulo 180 degrees. Inclination
is zero face-on and 90 degrees edge-on. `rebuild_galaxies.orientation()` converts
these measurements through the same ICRS-to-Galactic adapter used for star
positions, then into Unity's Z-X-Y Euler convention. Unity XZ is the Galactic
plane; its Y axis is Galactic north. The placement orbit plane is artificial
and must not be used to orient a galaxy.

| Galaxy | PA | Inclination | Adoption and limitations |
| --- | ---: | ---: | --- |
| Milky Way | Galactic plane | Galactic plane | Plane unchanged. Long bar adopted at 30 degrees to the Sun-center line; spiral phase remains approximate. |
| Andromeda / M31 | 38 degrees | 77 degrees | Representative main disk; outer warp is omitted. |
| Triangulum / M33 | 23 degrees | 54 degrees | Representative inner disk; PA uncertainty about 2 degrees and an outer warp. |
| NGC 300 | 111 degrees | 42 degrees | Published stellar-disk geometry replaces the catalog-summary PA of 114 degrees. |
| NGC 2403 | 124 degrees | 63 degrees | Published THINGS/CHAOS geometry replaces the generic axial-ratio estimate. |
| NGC 404 | about 80 degrees | about 11 degrees | Optical stellar PA; large-scale HI inclination used as a near-face-on proxy. Stellar tilt is uncertain; nuclear and outer gas components differ. |
| UGC 1382 | 45 degrees | about 46 degrees | Optical stellar-profile PA replaces the catalog value of 51 degrees. Larger gas disk and isophotes can have different geometry. |
| UGC 2885 | 90 degrees | 65 degrees | Retained SIMBAD PA and reviewed published inclination. |
| NGC 185 | 35 degrees | unknown | Retained projected spheroid, not a measured disk rotation axis. |
| NGC 147 | 25 degrees | unknown | Retained projected spheroid, not a measured disk rotation axis. |
| NGC 4889 | 75 degrees | unknown | Retained projected spheroid; true shape and depth are unknown. |
| NGC 6166 | 34 degrees | unknown | Retained projected spheroid; true shape and depth are unknown. |
| IC 1101 | 22 degrees | unknown | Retained SIMBAD PA; a single ellipsoid cannot reproduce radial isophote twists. |
| ESO 383-76 | 5 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |
| ESO 306-17 | 178 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |
| Holm 15A | 148 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |
| NGC 6173 | 141 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |
| NGC 2832 | 161 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |
| UGC 9799 | 34 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |
| NGC 1600 | 9 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |
| NGC 6086 | 3 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |
| NGC 4874 | 63 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |
| NGC 1132 | 139 degrees | unknown | Catalog projected stellar ellipsoid; see the [new batch notes](LargeEllipticalGalaxies.md). |

For spheroids, the renderer's local XZ ellipse faces Sol and reproduces the
catalog elongation. The internal zero-degree inclination is a rendering
construction, not evidence for a face-on physical disk. PA and inclination
alone also leave two possible near sides for a disk. The adopted side is a
convention, not a kinematic determination. Winding sense, individual arm phase,
and a unique three-dimensional rotation axis are not established for every
galaxy by these inputs. The game descriptions briefly state these limitations.

The other pre-existing galaxy bodies currently have no `GalacticDiskDefinition`
to orient; no new disks were added for them.

## Milky Way bar and legacy placement

SagittariusA's XZ plane was already correct. Its previous `rotationY = 75` and
local `barAngle = 25` combined to put the bar 50 degrees from the Sun-center
line. The yaw is now 55 degrees, retaining the local bar angle. GalacticDisk's
bar angle increases from local X towards Z, while positive Unity yaw turns X
towards negative Z: the resulting world bar angle is 25 - 55 = -30 degrees.
The end nearer Sol is at positive Galactic longitude. The adopted 30 degrees
lies within the published 28-33-degree long-bar range. This does not claim an
exact fit to the Milky Way's spiral arms, vertical structure, or warp.

Andromeda and Triangulum previously had zero argument of periapsis and anomaly
on their placement orbits, placing both in the Galactic plane at epoch zero,
regardless of the inclination field. Their orbits now encode the observed J2000
directions, with the existing distances of 2.40e22 m and 2.58e22 m retained.
Both use Sun as parent, `period = 3E300`, and `mode = OFF`. This is necessary for
their measured orientations to project correctly from Sol. Their existing
bodies and children are retained. Sun itself and SagittariusA's placement
orbit were not changed.

Andromeda's old SOI of 3e22 m exceeded its approximately 2.4e22-m distance from
Sol. Its SOI is now a 100-pc game core region, matching the generator's maximum
core size, so the new Sun-parent hierarchy does not enclose Sol in Andromeda's
SOI. This is a gameplay boundary, not a measured black-hole influence radius.
Triangulum's existing small core SOI is retained.

## Saved inputs and references

Generated galaxies use `data/galaxies.json`:
`rendering_overrides.<id>.inclination_deg` and `.position_angle_deg` override the
generic catalog geometry independently of photometric rendering parameters.
These reviewed values, sources, and notes survive source refreshes and normal
`python rebuild_galaxies.py` rebuilds. Raw catalog measurements are retained.

The three legacy disks are hand-authored configs. Their adopted measurements,
paths, Milky Way bar convention, and preserved placement distances are recorded
in `data/legacy_disk_orientations.json`. The galaxy importer continues to
exclude those bodies. Their configurations are checked by the orientation
tests, and `rebuild_galaxies.orientation()` can recompute their external-galaxy
Euler angles if the coordinate adapter changes.

- [Wegg, Gerhard & Portail (2015), Milky Way long bar](https://arxiv.org/abs/1504.01401): bar angle range 28-33 degrees.
- [Zhang et al. (2024), M31 rotation curve and adopted geometry](https://academic.oup.com/mnras/article/528/2/2653/7512223): center, PA 38 degrees, inclination 77 degrees.
- [Corbelli et al. (2019), M33 molecular clouds](https://doi.org/10.1051/0004-6361/201834437): PA 23 +/- 2 degrees and inner disk inclination 54 degrees. [SIMBAD](https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=M33) supplies its J2000 center.
- [Gogarten et al. (2010), NGC 300 stellar disk](https://doi.org/10.1088/0004-637X/712/2/858): PA 111 degrees, inclination 42 degrees.
- [Rogers et al. (2021), CHAOS VI Table 1](https://arxiv.org/abs/2105.03437): NGC 2403 PA 124 degrees, inclination 63 degrees, adopting de Blok et al. (2008).
- [del Rio, Brinks & Cepa (2004), NGC 404 HI](https://arxiv.org/abs/astro-ph/0403467): near-face-on inner HI geometry and outer warp. [Seth et al. (2010)](https://arxiv.org/abs/1003.0680): optical stellar isophotes near PA 80 degrees. The chosen single-plane stellar proxy is explicitly approximate.
- [Hagen et al. (2016), UGC 1382 Section III.1](https://arxiv.org/abs/1607.02147): optical profile PA 45 degrees, axis ratio 0.69; retained approximate 46-degree inclination is consistent with this flattening.
- [Carvalho et al. (2024), UGC 2885 Table 1](https://arxiv.org/html/2410.16467v1): retained 65-degree inclination.

The remaining projected PAs are traceable to the saved SIMBAD responses in
`data/galaxy_sources/`. Their three-dimensional shape is not inferred from PA.

## Verification

The tests independently project every shipped disk's Euler basis onto celestial
north/east and recover the adopted PA and inclination. They check complete disk
coverage, static world axes, the Milky Way plane and near bar end, and the two
corrected legacy heliocentric placement directions, distances, and small core SOIs. Generated
galaxy and star rebuild checks are also run. These are geometry and consistency
checks, not a live KSP play test.
