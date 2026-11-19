# Fifty nearby galaxies

This batch adds 50 new galaxies, bringing the generated galaxy catalog to 100
(103 GalacticDisk definitions including the three legacy galaxy bodies).
Selection prioritizes nominal heliocentric distance, then supported morphology
and sufficient measurements. It uses the Local Volume Database's confirmed-galaxy
sample and a small set of explicit literature exclusions. It is not a claim to
include every object ever proposed as a nearby galaxy.

Only three selected objects fall within 100,000 light-years: Segue 1, Hydrus I,
and Triangulum II. The fiftieth is Andromeda XXIV at about 1.98 million light-years.
All fifty are dwarf spheroidals: the closest eligible objects are smooth dwarf
galaxies, while the nearby major spirals are already present. Distance takes
priority over selecting another, much more distant spiral. Irregular, ring, and
merging systems remain outside the supported import families.

## Source and selection

Measurements are pinned to [Local Volume Database v1.1.2, released November 4,
2026](https://github.com/apace7/local_volume_database/releases/tag/v1.1.2).
The [database paper](https://arxiv.org/abs/2411.07424) and
[field documentation](https://local-volume-database.readthedocs.io/en/latest/usage.html)
describe the compilation. The importer checks the release CSV's published SHA256:

`6a9e03ba53f08f1cf1c07ca21004286a71a7b8c4edcec108a11ec484d1141ffb`

The original CSV, version, retrieval time and checksum are saved in
`data/galaxy_sources/lvdb_nearby_snapshot.json`. Every new galaxy also has its own
raw row, reviewed metadata, field references, and assumptions in the linked JSON
record below. `data/nearby_galaxy_selection.json` records the distance ordering
and reasons for skipped candidates before the last addition. Later imports
retain earlier audits under `previous_selections` in the same file.

Eligibility requires confirmed-real and confirmed-galaxy flags, a supported
smooth morphology, heliocentric distance, coordinates, half-light size,
flattening, position angle, V magnitude, stellar mass estimate and radial
velocity. A tightly constrained, nearly circular ellipticity upper limit can
stand in for flattening; Crater II uses this explicitly documented exception.
The importer does not invent missing colors, sizes, or radial velocities.

Additional reviewed exclusions are:

- Draco II: [current galaxy/cluster classification debate](https://arxiv.org/abs/2610.28738).
- Carina III: [four spectroscopic members do not establish its nature](https://arxiv.org/abs/1802.06810).
- Willman 1: [classification remains uncertain](https://arxiv.org/abs/2602.20272).
- Sagittarius dwarf spheroidal: actively disrupting and too extended for this
  bounded, smooth template. Its tidal debris is not approximated as a disk.

Ursa Major III/UNIONS 1 is classified as a star cluster in this release and fails
the galaxy filter. Other candidates with incomplete geometry or unsupported
morphology are recorded in the selection audit.

## Additions and measured geometry

Distances below are nominal values, not exact positions. Asymmetric catalog
distance errors and original reference identifiers are retained in the JSON.
`r_half` is the measured projected elliptical semi-major half-light radius in
arcminutes. The model diameter is an inferred Gaussian extent, not a measured
tidal or isophotal boundary. PA is the projected major-axis position angle,
north through east, modulo 180 degrees. `q` is projected minor/major axis ratio.
Crater II's PA of zero is only a convention for its assumed circular projection.

| Galaxy / saved measurements | Distance (kpc) | Distance (ly) | r_half (arcmin) | Model diameter (arcmin) | PA (deg) | q |
|---|---:|---:|---:|---:|---:|---:|
| [Segue 1](../data/galaxy_sources/lvdb_segue_1.json) | 22.91 | 74,722 | 3.620 | 18.447 | 77.00 | 0.670 |
| [Hydrus I](../data/galaxy_sources/lvdb_hydrus_1.json) | 27.54 | 89,823 | 7.420 | 37.811 | 97.00 | 0.790 |
| [Triangulum II](../data/galaxy_sources/lvdb_triangulum_2.json) | 28.44 | 92,759 | 2.500 | 12.740 | 73.00 | 0.700 |
| [Reticulum II](../data/galaxy_sources/lvdb_reticulum_2.json) | 31.62 | 103,131 | 6.300 | 32.104 | 68.00 | 0.400 |
| [Ursa Major II](../data/galaxy_sources/lvdb_ursa_major_2.json) | 34.67 | 113,078 | 13.800 | 70.322 | 104.00 | 0.440 |
| [Segue 2](../data/galaxy_sources/lvdb_segue_2.json) | 36.48 | 118,982 | 3.760 | 19.161 | 164.00 | 0.780 |
| [Carina II](../data/galaxy_sources/lvdb_carina_2.json) | 37.39 | 121,950 | 8.690 | 44.283 | 170.00 | 0.660 |
| [Bootes II](../data/galaxy_sources/lvdb_bootes_2.json) | 41.69 | 135,975 | 3.170 | 16.154 | 112.00 | 0.750 |
| [Coma Berenices](../data/galaxy_sources/lvdb_coma_berenices_1.json) | 42.27 | 137,866 | 5.640 | 28.741 | 123.00 | 0.630 |
| [Pictor II](../data/galaxy_sources/lvdb_pictor_2.json) | 44.87 | 146,346 | 2.800 | 14.269 | 55.00 | 0.780 |
| [Tucana IV](../data/galaxy_sources/lvdb_tucana_4.json) | 46.99 | 153,261 | 9.300 | 47.392 | 27.00 | 0.610 |
| [Bootes III](../data/galaxy_sources/lvdb_bootes_3.json) | 48.53 | 158,284 | 40.600 | 206.842 | 99.00 | 0.670 |
| [Tucana V](../data/galaxy_sources/lvdb_tucana_5.json) | 54.95 | 179,223 | 2.100 | 10.701 | 29.00 | 0.490 |
| [Grus II](../data/galaxy_sources/lvdb_grus_2.json) | 55.21 | 180,071 | 6.800 | 34.652 | 6.00 | 0.750 |
| [Tucana II](../data/galaxy_sources/lvdb_tucana_2.json) | 56.23 | 183,398 | 12.890 | 65.685 | 107.00 | 0.610 |
| [Bootes I](../data/galaxy_sources/lvdb_bootes_1.json) | 66.37 | 216,470 | 14.730 | 75.061 | 3.12 | 0.572 |
| [Eridanus IV](../data/galaxy_sources/lvdb_eridanus_4.json) | 69.82 | 227,722 | 3.240 | 16.511 | 75.00 | 0.740 |
| [Ursa Minor](../data/galaxy_sources/lvdb_ursa_minor_1.json) | 70.15 | 228,799 | 18.300 | 93.251 | 50.00 | 0.450 |
| [DELVE 2](../data/galaxy_sources/lvdb_delve_2.json) | 71.12 | 231,962 | 1.040 | 5.300 | 74.00 | 0.970 |
| [Horologium I](../data/galaxy_sources/lvdb_horologium_1.json) | 79.43 | 259,066 | 1.610 | 8.204 | 44.00 | 0.730 |
| [Draco](../data/galaxy_sources/lvdb_draco_1.json) | 81.55 | 265,981 | 10.840 | 55.239 | 92.37 | 0.734 |
| [Phoenix II](../data/galaxy_sources/lvdb_phoenix_2.json) | 83.18 | 271,297 | 1.500 | 7.644 | 147.00 | 0.560 |
| [Sculptor](../data/galaxy_sources/lvdb_sculptor_1.json) | 83.95 | 273,808 | 11.170 | 56.920 | 92.00 | 0.670 |
| [Aquarius III](../data/galaxy_sources/lvdb_aquarius_3.json) | 85.51 | 278,896 | 2.100 | 10.701 | 119.00 | 0.530 |
| [Sextans](../data/galaxy_sources/lvdb_sextans_1.json) | 85.90 | 280,168 | 22.800 | 116.178 | 52.00 | 0.730 |
| [Ursa Major I](../data/galaxy_sources/lvdb_ursa_major_1.json) | 97.27 | 317,252 | 8.310 | 42.347 | 67.00 | 0.410 |
| [Bootes V](../data/galaxy_sources/lvdb_bootes_5.json) | 101.86 | 332,223 | 0.760 | 3.873 | 18.00 | 0.800 |
| [Carina](../data/galaxy_sources/lvdb_carina_1.json) | 105.58 | 344,356 | 10.100 | 51.468 | 60.00 | 0.640 |
| [Aquarius II](../data/galaxy_sources/lvdb_aquarius_2.json) | 107.65 | 351,107 | 5.100 | 25.989 | 121.00 | 0.610 |
| [Leo VI](../data/galaxy_sources/lvdb_leo_6.json) | 111.17 | 362,588 | 4.170 | 21.250 | 63.00 | 0.460 |
| [Crater II](../data/galaxy_sources/lvdb_crater_2.json) | 116.57 | 380,200 | 31.200 | 158.969 | 0.00 | 1.000 |
| [Centaurus I](../data/galaxy_sources/lvdb_centaurus_1.json) | 117.71 | 383,919 | 2.600 | 13.249 | 10.00 | 0.640 |
| [Antlia II](../data/galaxy_sources/lvdb_antlia_2.json) | 124.17 | 404,988 | 104.600 | 532.134 | 156.00 | 0.400 |
| [Grus I](../data/galaxy_sources/lvdb_grus_1.json) | 126.47 | 412,490 | 4.160 | 21.199 | 153.00 | 0.560 |
| [Hercules](../data/galaxy_sources/lvdb_hercules_1.json) | 130.62 | 426,025 | 5.630 | 28.690 | 107.00 | 0.310 |
| [Fornax](../data/galaxy_sources/lvdb_fornax_1.json) | 142.56 | 464,969 | 19.900 | 101.403 | 42.70 | 0.710 |
| [Hydra II](../data/galaxy_sources/lvdb_hydra_2.json) | 151.36 | 493,670 | 2.130 | 10.854 | 171.00 | 0.700 |
| [Leo IV](../data/galaxy_sources/lvdb_leo_4.json) | 151.36 | 493,670 | 2.540 | 12.944 | 152.00 | 0.830 |
| [Canes Venatici II](../data/galaxy_sources/lvdb_canes_venatici_2.json) | 159.96 | 521,720 | 1.520 | 7.746 | 9.00 | 0.600 |
| [Leo V](../data/galaxy_sources/lvdb_leo_5.json) | 169.04 | 551,335 | 1.000 | 5.096 | 109.00 | 0.570 |
| [Columba I](../data/galaxy_sources/lvdb_columba_1.json) | 182.81 | 596,246 | 2.200 | 11.211 | 24.00 | 0.700 |
| [Pisces II](../data/galaxy_sources/lvdb_pisces_2.json) | 182.81 | 596,246 | 1.340 | 6.829 | 98.00 | 0.630 |
| [Leo II](../data/galaxy_sources/lvdb_leo_2.json) | 208.93 | 681,439 | 2.520 | 12.842 | 38.00 | 0.930 |
| [Canes Venatici I](../data/galaxy_sources/lvdb_canes_venatici_1.json) | 210.86 | 687,733 | 7.120 | 36.283 | 80.00 | 0.560 |
| [Pegasus III](../data/galaxy_sources/lvdb_pegasus_3.json) | 214.78 | 700,519 | 1.670 | 8.510 | 83.00 | 0.630 |
| [Leo I](../data/galaxy_sources/lvdb_leo_1.json) | 258.23 | 842,234 | 3.650 | 18.600 | 78.00 | 0.700 |
| [Eridanus II](../data/galaxy_sources/lvdb_eridanus_2.json) | 369.83 | 1,206,224 | 2.310 | 11.772 | 72.60 | 0.520 |
| [Andromeda XVI](../data/galaxy_sources/lvdb_andromeda_16.json) | 517.61 | 1,688,218 | 1.000 | 5.096 | 98.00 | 0.710 |
| [Cassiopeia II](../data/galaxy_sources/lvdb_cassiopeia_2.json) | 559.76 | 1,825,693 | 1.500 | 7.644 | 110.00 | 0.570 |
| [Andromeda XXIV](../data/galaxy_sources/lvdb_andromeda_24.json) | 608.14 | 1,983,487 | 2.600 | 13.249 | 90.00 | 0.900 |

## How the data become game objects

Positions use ICRS/J2000 coordinates and heliocentric distances, converted into
the existing Galactic game axes. Every barycenter is placed on a hidden,
Sun-parented, circular orbit with `period = 3E300` and `mode = OFF`. Stock Sun is
unchanged. Each core SOI is limited to 5-100 parsecs and less than one tenth of
the nominal model radius. Configs are files directly inside
`RealGalacticSystems/Systems`, named after the catalog IDs above.

A projected Gaussian has `r_half = sigma * sqrt(2 ln 2)`. The nominal boundary
for these objects is three sigma. The importer uses angular tangents to obtain
the physical scale and the compiler sets the normalized Gaussian sigma to
`100 / 3`, with the second sky axis multiplied by the measured axis ratio.
The measured half-light size is therefore preserved in this model. Existing
spheroids retain their original 3.5-sigma convention and generated output.

These are pressure-supported systems. Their measured elongation and PA define
the apparent shape; they do not establish a disk inclination or a physical spin
axis. A triaxial Gaussian is only an assumed three-dimensional continuation.
All rendered structures remain static, independent of their position-only
orbits. Their observed light is already from the emission epoch; descriptions
report the approximate local `distance / c` lookback without inventing another
epoch of structural evolution. Radial velocity divided by c is a local Doppler
redshift proxy, not the distance estimator.

The source V magnitudes are extinction-corrected. They are used directly when B
is unavailable, with no fabricated B-V color. Foreground extinction is not
reconstructed. The renderer's existing minimum brightness and photographic
exposure mapping remain in use: these faint objects are display approximations,
not a calibrated prediction of naked-eye visibility. Old-population colors are
template estimates. Stellar mass is `10**log_mass` from the database's assumed
V-band mass/light ratio of two; it excludes the dark halo and is only a game
point-mass proxy, not a central black-hole or measured dynamical mass.

Every description includes these modeling uncertainties. Object-specific notes
also flag Triangulum II's binary/velocity uncertainties, possible tidal structure
in Bootes III, Ursa Major II and Antlia II, sparse DELVE 2 measurements, unresolved
Grus II internal dispersion, and Crater II's assumed circular projection.
Tidal streams, foreground stars, gas, and resolved stellar population histories
are not reconstructed.

## Rebuild or import more

To regenerate existing objects from saved data and templates:

```powershell
python rebuild_galaxies.py
python rebuild_galaxies.py --check
```

The standard-library-only importer selects the nearest **new** eligible
spheroidal galaxies. Running it again after this batch selects further objects,
not these same fifty. Review a proposed next batch before saving it:

```powershell
python fetch_nearby_galaxies.py --offline --count 5 --dry-run
python fetch_nearby_galaxies.py --offline --count 5
```

Omit `--offline` to download the same pinned, checksum-verified release. A future
release requires updating the CSV checksum and matching reviewed metadata
together. This importer complements the existing SIMBAD/VizieR importer for
regular spirals and larger galaxies; it does not add unsupported irregulars.

## Verification

All 54 automated tests pass, including source integrity, confirmation filters,
nearest-first selection, unit conversions, preservation of measured half-light
size and apparent shape, original-source replay, write-free dry runs, and
preservation of existing objects and selection history during another import.
All 100 generated galaxy configs and 150 generated star configs match their
saved data and templates. Projection checks cover all 103 disk definitions.

All fifty additions were rendered and visually inspected with the actual local
GalacticDisk C# renderer and packaged shader in an isolated Unity 2019.4 project,
looking from Sol along each galaxy's sky direction. These previews use KSP API
substitutes and exclude game foreground, bloom and user saves. They validate
renderability and model geometry; they are not a live KSP play test or absolute
photometric calibration. They show faint smooth ellipses, not spiral arms.

The original fifty generated galaxies, all legacy galaxy configs and disk
orientations, all star configs, stock Sun, and README are preserved.
