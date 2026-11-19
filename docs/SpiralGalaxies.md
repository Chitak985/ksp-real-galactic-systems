# Thirty spiral galaxies

Added on 2026-11-18 through the existing importer. The saved catalog now contains
50 generated galaxies. This batch contains 17 ordinary spirals, seven weakly
barred spirals, and six barred spirals. Configs are individual files directly
under `RealGalacticSystems/Systems`.

The selection emphasizes actual spiral stellar disks with complete positions,
photometry, dimensions, distance estimates, and orientation inputs. It includes
face-on, inclined, and edge-on examples; viewing a real spiral edge-on hides
its arms. The selected sources are SIMBAD, PHANGS-ALMA, and SPARC. Dedicated
irregular, collisional-ring, and ongoing-merger shapes are not introduced.

## Saved galaxies and orientations

PA is east of celestial north modulo 180 degrees; inclination is zero face-on
and 90 edge-on. Published inclinations are available for 28 objects. NGC 5908 and
NGC 5965 instead use the existing axial-ratio estimate with assumed intrinsic
thickness 0.15. A single plane omits warps, and neither PA nor inclination alone
identifies which side is nearer or fixes the exact arm/bar phase.

Distances in the table are game placement distances. Local adopted distances
are used directly; the four higher-redshift objects use the existing comoving
cosmological placement. Stellar diameter always uses angular-diameter distance.
Catalog isophotes differ by bandpass and depth: these are conservative stellar
outlines, not gas diameters or an identical-isophote size ranking.

| Galaxy | Template | Placement distance (Mpc) | Stellar diameter (kpc) | PA (degrees) | Inclination (degrees) | Inclination source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| NGC 4945 | barred spiral | 3.47 | 20.6 | 43.8 | 90.0 | measured |
| NGC 7793 | spiral | 3.62 | 10.5 | 110.0 | 50.0 | measured |
| NGC 253 | weakly barred spiral | 3.70 | 31.0 | 52.5 | 75.0 | measured |
| NGC 247 | weakly barred spiral | 3.71 | 18.4 | 167.4 | 76.4 | measured |
| M 83 (NGC 5236) | weakly barred spiral | 4.89 | 19.6 | 45.0 | 24.0 | measured |
| NGC 6503 | spiral | 6.26 | 12.3 | 123.0 | 74.0 | measured |
| NGC 3621 | spiral | 7.06 | 22.5 | 163.8 | 65.8 | measured |
| IC 5332 | spiral | 9.01 | 21.3 | 74.4 | 26.9 | measured |
| M 74 (NGC 628) | spiral | 9.84 | 19.2 | 20.7 | 8.9 | measured |
| NGC 891 | spiral | 9.91 | 35.5 | 22.0 | 90.0 | measured |
| NGC 3511 | spiral | 13.94 | 25.0 | 76.8 | 75.1 | measured |
| NGC 2997 | weakly barred spiral | 14.06 | 40.0 | 108.1 | 33.0 | measured |
| NGC 7814 | spiral | 14.40 | 28.3 | 134.0 | 90.0 | measured |
| NGC 7331 | spiral | 14.70 | 44.8 | 171.0 | 75.0 | measured |
| M 100 (NGC 4321) | weakly barred spiral | 15.21 | 27.9 | 156.2 | 38.5 | measured |
| NGC 5033 | spiral | 15.70 | 28.7 | 177.0 | 66.0 | measured |
| NGC 4535 | weakly barred spiral | 15.77 | 29.4 | 179.7 | 44.7 | measured |
| NGC 5907 | spiral | 17.30 | 77.9 | 155.0 | 88.0 | measured |
| NGC 1566 | weakly barred spiral | 17.69 | 27.5 | 34.7 | 29.5 | measured |
| NGC 3917 | spiral | 18.00 | 46.3 | 77.0 | 79.0 | measured |
| NGC 3949 | spiral | 18.00 | 14.6 | 119.0 | 55.0 | measured |
| NGC 4217 | spiral | 18.00 | 40.1 | 49.0 | 86.0 | measured |
| NGC 1300 | barred spiral | 18.99 | 37.8 | 98.0 | 31.8 | measured |
| NGC 1672 | barred spiral | 19.40 | 38.5 | 134.3 | 42.6 | measured |
| NGC 1559 | barred spiral | 19.44 | 26.8 | 64.5 | 65.4 | measured |
| NGC 1365 | barred spiral | 19.57 | 43.0 | 21.1 | 55.4 | measured |
| NGC 5908 | spiral | 49.54 | 51.2 | 154.0 | 67.6 | axial-ratio estimate |
| NGC 5965 | barred spiral | 49.99 | 64.3 | 53.0 | 82.6 | axial-ratio estimate |
| UGC 128 | spiral | 67.50 | 40.6 | 65.0 | 57.0 | measured |
| NGC 801 | spiral | 86.08 | 31.4 | 150.0 | 80.0 | measured |

Every addition is one invisible barycenter on a circular Sun-parented placement
orbit with `period = 3E300`, `mode = OFF`, a small core SOI, and an independent,
static world-space disk orientation. The observed light already represents its
emission epoch; it is not rewound a second time. Game descriptions mention
uncertain distances/masses, poorly constrained inclinations where applicable,
and omitted small-scale structures.

## Data and reviewed approximations

- [PHANGS-ALMA, Leroy et al. (2021), Tables 3 and 4](https://arxiv.org/html/2104.07739): compiled local distances, representative disk PA/inclination with uncertainties, and photometric stellar-mass estimates. Near-face-on inclinations can have large errors; errors and original receding-side PAs are saved. Some distances use group/flow estimates and are not labeled redshift-independent.
- [SPARC, Lelli et al. (2016), original Table 1](https://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt): adopted disk inclinations, Hubble stages, and local distances with method/error flags. NGC 801 and UGC 128 retain the generator's redshift-based cosmological distance instead of the SPARC flow distances. NGC 5033 uses the published local flow-model estimate with its large uncertainty.
- [SIMBAD](https://simbad.cfa.harvard.edu/simbad/): J2000 astrometry, observed redshift, angular stellar dimensions, photometry, and sky PA where PHANGS does not supersede it. The original response and field references are retained for each galaxy.

PHANGS stellar masses are photometric estimates, not dynamical black-hole
masses. Other galaxies use the existing Ks stellar mass-to-light estimate.
UGC 128 lacks Ks in this query, so its stellar mass is estimated from the saved
SPARC 3.6-micron luminosity using mass-to-light ratio 0.5 solar. The result is
6.01 billion solar masses. Gravity remains a game point-mass proxy; it does not
reproduce a distributed stellar/dark-matter potential.

NGC 1300 and NGC 2997 have `(rs)` details in the catalog classifications. They
are reviewed as ordinary barred/weakly barred spiral approximations, consistent
with their [NASA](https://science.nasa.gov/asset/hubble/spiral-galaxy-ngc-1300/)
and [ESO](https://www.eso.org/public/images/ngc_2997/) descriptions. Small
pseudoring/nuclear-spiral details are explicitly omitted; no ring template or
collisional ring system is added. NGC 1672's numeric Hubble stage does not encode
its strong bar, so its [observed barred structure](https://science.nasa.gov/asset/hubble/hubbles-view-of-barred-spiral-galaxy-ngc-1672/)
selects the barred template. NGC 7814's dominant bulge, NGC 5907's faint streams,
and nuclear starburst/outflow structures are not detailed reconstructions.
Arm pitch, winding, bar angle/phase, dust clumps, and color palettes remain
renderer estimates. The templates reproduce galaxy families rather than
individual photographic textures.

Barred cases use saved bar lengths and reduced round-bulge scales so the bar is
visibly elongated. Those visual values are estimates, not additional measured
galaxy dimensions. Requested catalog identifiers remain stable even when
SIMBAD prefers a Messier or MCG alias; raw aliases are retained as provenance.

## Rebuild and refresh

Reviewed inputs and references are in `data/spiral_galaxy_adoptions.json`.
`data/galaxies.json` preserves their orientation refinements. Original individual
SIMBAD responses and the source survey tables are saved under
`data/galaxy_sources`. Rebuilding needs no network:

```powershell
python rebuild_galaxies.py
python rebuild_galaxies.py --check
```

The nearby-catalog service returned incomplete responses for many candidates.
An explicit `--simbad-only` import mode now supports reviewed alternate-source
data without issuing that query. It records the omission in provenance and
still rejects missing required values. It never turns a failed query into an
empty successful catalog match. Default imports retain their strict behavior.
To refresh an individual entry while preserving reviewed fields:

```powershell
python fetch_galaxies.py "NGC 1365" --refresh --simbad-only --overrides data/spiral_galaxy_adoptions.json
```

Add `--offline` to replay saved data instead of requesting fresh data. Survey
values in the adoption file are a documented snapshot; refreshing SIMBAD does
not silently replace those reviewed distance and orientation measurements.
The `adopted_local` distance kind distinguishes these compiled local estimates
from strictly redshift-independent distances; both use an approximate D/c
lookback. Higher-redshift placement retains the existing cosmology.

The previous 20 galaxy records, their configurations and rendering refinements,
all stars and legacy galaxy files, Sun, README, and Git state are preserved.

## Verification

All 45 Python tests pass. Tests cover offline source replay, explicit omission
of the unavailable nearby query, local distance validation, survey PA/inclination
and mass units, and sky projection for all 53 disks. All 50 generated galaxy
configs and 150 star configs match their saved inputs. All thirty new galaxies
were rendered and visually inspected using the actual GalacticDisk renderer
and packaged shaders in an isolated Unity preview looking from Sol with
celestial north up. The barred cases were checked again after visual tuning.
These checks use KSP API substitutes, no Milky Way foreground, and no user save;
they are not a live KSP play test or a photographic reconstruction.
