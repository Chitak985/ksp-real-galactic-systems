# Nine hundred additional galaxies

This batch adds exactly **900** real galaxies: **34** remaining eligible nearby
LVDB systems and **866** SIMBAD galaxies, bringing the generated catalog to
**1,000**, or **1,003** disk definitions including the three legacy galaxy bodies.
The pre-existing 100 galaxies, all stars, Sun, README and Git state are preserved.

## Selection and sources

The nearby additions use the same pinned [LVDB v1.1.2 release](https://github.com/apace7/local_volume_database/releases/tag/v1.1.2)
and reviewed filters described in [the previous batch](NearbyGalaxies.md).
They span approximately 2.06-12.23 million light-years. The 866 farther additions
use [SIMBAD](https://simbad.harvard.edu/simbad/) measurements retrieved on
2026-11-19, ordered by redshift and stable SIMBAD object ID. They span approximately
154.7-372.3 million light-years in comoving placement distance, with light-travel
times of approximately 153.9-367.6 million years. These two ranges intentionally
use different distance definitions; comoving distance divided by c is not the
cosmological lookback time.

SIMBAD is an inhomogeneous literature compilation. This is a reproducible
selection of well-populated records, not a volume-complete or unbiased census,
and the gap between these source samples does not mean no galaxies exist there.
Original redshift, morphology and angular-dimension bibcodes are retained with
each raw row and adopted entry; B/K photometry is retained as database values.
The [SIMBAD paper](https://arxiv.org/abs/astro-ph/0002110) describes the database.
SIMBAD data are available under ODbL; preserve source attribution and consult
its linked database terms when redistributing the data compilation.

The exact bulk ADQL query, original JSON response, parsed rows, retrieval time,
URL and SHA256 are saved in `data/galaxy_sources/simbad_bulk_snapshot.json`.
The importer verifies that the response hash and parsed rows agree. This hash
checks the saved response's integrity; it is not a publisher-signed release hash.
Every accepted galaxy also has an individual saved raw record. The ordered
selection and reasons for 967 rejected candidates encountered before the last
accepted object are recorded in `data/bulk_galaxy_selection.json`.

The bulk query examines the first 4,000 records with object type G, redshift
0.01-0.1, positive major/minor dimensions, position angle, morphology and B/K
photometry. Additional acceptance gates require:

- Explicitly spectroscopic redshift nature (`s`, `se` or `sa`), with A/B/C quality.
- A/B/C angular-dimension quality and original dimension, morphology and redshift references.
- A supported regular morphology, without question-mark, colon or slash ambiguity.
- Finite coordinates, photometry and geometry; plausible inferred radius
  (100 pc-200 kpc), stellar mass (10^5-10^14 solar masses), and axis ratio >=0.05.
- No existing identifier match, duplicate SIMBAD object ID, duplicate body name,
  or object within five arcseconds of an existing/accepted galaxy without review.

Although any galaxy type was allowed for this request, enough measured regular
systems were available. No new irregular, merging or ring template was needed.
Those structures are not silently substituted with smooth models. Bar presence
comes from the catalog morphology; detailed bar and spiral geometry remain
template estimates.

## Composition

| Family | Added |
|---|---:|
| Elliptical | 46 |
| Compact elliptical | 1 |
| Dwarf elliptical | 1 |
| Dwarf spheroidal | 32 |
| Lenticular | 76 |
| Barred lenticular | 4 |
| Spiral | 642 |
| Weakly barred spiral | 25 |
| Barred spiral | 73 |

## Positioning, appearance and uncertainties

Every new invisible barycenter uses a hidden, circular Sun-parented orbit with
`period = 3E300` and `mode = OFF`, purely for placement. Files are directly under
`RealGalacticSystems/Systems`, not individual galaxy directories. Core SOIs use
the existing 5-100 pc bounds and remain below one tenth of the nominal galaxy
radius. A pairwise check found no overlapping SOIs involving these additions
among the 1,000 generated galaxy barycenters.

The farther sample uses the existing flat FLRW cosmology (H0=67.4 km/s/Mpc,
Omega_m=0.315, Omega_r=0.000092). Comoving distance sets position; angular-diameter
distance sets physical/rendered extent so observed angular size is preserved;
luminosity distance sets the Ks luminosity estimate. Low-redshift peculiar
velocities, extinction, heterogeneous photometric apertures and missing dark
matter add uncertainty. No precision percentage is claimed for these estimates.

The observed morphology already shows emitted light from the past; it is not
evolved backward a second time. Approximate lookback times are included in each
description. Stellar mass assumes Ks mass/light=1 solar, excludes dark halos,
and is a game gravity proxy, not a measured black-hole mass. Colors are redshifted
template palettes, not reconstructed spectra. Apparent PA and flattening are
measured; inclination, near-side choice, depth, spiral phase, warp and detailed
structure are uncertain. Nearby spheroids keep their documented half-light and
mass/light conventions, and apparent elongation does not establish a spin axis.
Each description briefly states the applicable uncertainties.

To limit the added mesh cost, these 900 objects each use **2,000 representative
mesh stars**, saved as an explicit `rendering_overrides` parameter. This totals
1.8 million additional mesh stars instead of 27 million with the default 30,000
per object. It is a visual sampling budget, not a physical star count. Existing
objects and templates keep their counts. The diffuse volume, geometry and
measured placement are unchanged. This still adds 900 active bodies and volumes;
startup time, memory and frame rate need a live KSP test at this scale. No renderer
or mod DLL has been modified, and no automatic distance LOD is claimed.

## Rebuild and future imports

Rebuild the installed catalog from the saved templates and data:

```powershell
python rebuild_galaxies.py
python rebuild_galaxies.py --check
```

The new standard-library-only importer adds **new** galaxies, preserving existing
adopted entries and configs. The exact 900-galaxy selection can be replayed by
removing just this batch from a copy of the catalog and using its saved snapshots.
For a future batch, inspect the proposed additions first:

```powershell
python fetch_bulk_galaxies.py --offline --count 50 --dry-run
python fetch_bulk_galaxies.py --offline --count 50
```

Without `--offline`, it re-queries the same selection bounds from live SIMBAD.
The query is capped at 4,000 rows; failure to find enough new qualified objects
stops before writes. Future broadening requires an explicit query change and
source review. Existing audits are retained under `previous_selections`.
The earlier nearby-only selection audit is preserved as its original batch
record; the 34 new nearby objects are tracked by this bulk batch's audit.

## Verification

All 60 automated tests pass. Coverage includes snapshot integrity, raw-record
replay, scientific quality filters, duplicate protection, frozen placement,
angular-size and sky-orientation projections, bounded mesh-star counts, template
propagation, and offline dry runs that leave files unchanged. Existing nearby
import tests now explicitly leave a known fixture available, so they continue
to test import behavior after this batch exhausts the eligible LVDB sample.
All 1,000 generated galaxy configs and all 150 generated star configs match the
saved data and templates. Orientation checks include all 1,003 disk definitions.

Twenty-eight representative additions, covering all nine families and their
angular-size/flattening extremes, were rendered using the current GalacticDisk
C# source and its packaged shaders in isolated Unity 2019.4 previews. Both
Sol-direction angular views and enlarged shape views were visually inspected.
They show the expected smooth spheroids, inclined disks, bars and spiral arms.
These checks use KSP API substitutes; they do not load a user save or validate
all 1,000 galaxies simultaneously in KSP. Resource and gameplay behavior at that
scale remain unmeasured. Original game configs were compared byte for byte.

## Full addition list

Distances are nominal estimates. Angular major axes are catalog diameters for
the farther objects and inferred Gaussian extents for the nearby spheroids.
PA is north through east, modulo 180 degrees; q is apparent minor/major ratio.
Links open individual source records with measurement references and assumptions.

| Galaxy / saved source | Family | Placement (Mpc) | Lookback (Myr) | z | Major axis (arcmin) | q | PA (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| [Andromeda X](../data/galaxy_sources/lvdb_andromeda_10.json) | dwarf spheroidal | 0.6310 | 2.058 | -0.0005473787 | 5.606 | 0.900 | 30.00 |
| [Andromeda II](../data/galaxy_sources/lvdb_andromeda_02.json) | dwarf spheroidal | 0.6668 | 2.175 | -0.0006417773 | 27.008 | 0.840 | 31.00 |
| [Andromeda IX](../data/galaxy_sources/lvdb_andromeda_09.json) | dwarf spheroidal | 0.7015 | 2.288 | -0.0006984832 | 10.192 | 1.000 | 41.00 |
| [Perseus I](../data/galaxy_sources/lvdb_perseus_1.json) | dwarf spheroidal | 0.7047 | 2.298 | -0.001087419 | 7.134 | 0.910 | 122.00 |
| [Andromeda XXIX](../data/galaxy_sources/lvdb_andromeda_29.json) | dwarf spheroidal | 0.7112 | 2.320 | -0.0006484486 | 7.083 | 0.710 | 55.00 |
| [Andromeda XII](../data/galaxy_sources/lvdb_andromeda_12.json) | dwarf spheroidal | 0.7178 | 2.341 | -0.001858286 | 9.173 | 0.390 | 16.00 |
| [Andromeda III](../data/galaxy_sources/lvdb_andromeda_03.json) | dwarf spheroidal | 0.7211 | 2.352 | -0.001148461 | 10.192 | 0.410 | 140.00 |
| [Andromeda XX](../data/galaxy_sources/lvdb_andromeda_20.json) | dwarf spheroidal | 0.7413 | 2.418 | -0.001521719 | 2.038 | 0.890 | 90.00 |
| [Andromeda XXIII](../data/galaxy_sources/lvdb_andromeda_23.json) | dwarf spheroidal | 0.7447 | 2.429 | -0.0007928819 | 27.518 | 0.590 | 138.00 |
| [Andromeda XXVIII](../data/galaxy_sources/lvdb_andromeda_28.json) | dwarf spheroidal | 0.7447 | 2.429 | -0.001104431 | 7.032 | 0.580 | 35.04 |
| [Lacerta I](../data/galaxy_sources/lvdb_lacerta_1.json) | dwarf spheroidal | 0.7447 | 2.429 | -0.0006617912 | 15.950 | 0.590 | 116.00 |
| [Andromeda XV](../data/galaxy_sources/lvdb_andromeda_15.json) | dwarf spheroidal | 0.7482 | 2.440 | -0.001077412 | 6.625 | 0.760 | 38.00 |
| [Andromeda XI](../data/galaxy_sources/lvdb_andromeda_11.json) | dwarf spheroidal | 0.7516 | 2.451 | -0.001425987 | 3.058 | 0.810 | 54.00 |
| [Andromeda XXV](../data/galaxy_sources/lvdb_andromeda_25.json) | dwarf spheroidal | 0.7516 | 2.451 | -0.0003592485 | 13.759 | 0.970 | 164.00 |
| [Andromeda XXII](../data/galaxy_sources/lvdb_andromeda_22.json) | dwarf spheroidal | 0.7551 | 2.463 | -0.0004329662 | 4.586 | 0.390 | 114.00 |
| [Cetus](../data/galaxy_sources/lvdb_cetus_1.json) | dwarf spheroidal | 0.7551 | 2.463 | -0.0002798603 | 16.307 | 0.670 | 63.00 |
| [Andromeda V](../data/galaxy_sources/lvdb_andromeda_05.json) | dwarf spheroidal | 0.7586 | 2.474 | -0.00132525 | 8.153 | 0.740 | 54.00 |
| [Andromeda XVII](../data/galaxy_sources/lvdb_andromeda_17.json) | dwarf spheroidal | 0.7586 | 2.474 | -0.0008392473 | 7.134 | 0.500 | 110.00 |
| [Andromeda XIV](../data/galaxy_sources/lvdb_andromeda_14.json) | dwarf spheroidal | 0.7727 | 2.520 | -0.001603109 | 7.644 | 0.830 | 176.00 |
| [Andromeda XXI](../data/galaxy_sources/lvdb_andromeda_21.json) | dwarf spheroidal | 0.7727 | 2.520 | -0.001212172 | 20.893 | 0.640 | 139.00 |
| [M 32](../data/galaxy_sources/lvdb_m_032.json) | compact elliptical | 0.7727 | 2.520 | -0.0006637925 | 2.395 | 0.750 | 159.00 |
| [Andromeda I](../data/galaxy_sources/lvdb_andromeda_01.json) | dwarf spheroidal | 0.7762 | 2.532 | -0.001255202 | 19.874 | 0.720 | 30.00 |
| [Andromeda XXVI](../data/galaxy_sources/lvdb_andromeda_26.json) | dwarf spheroidal | 0.7870 | 2.567 | -0.0008726037 | 5.096 | 0.650 | 50.00 |
| [Cassiopeia III](../data/galaxy_sources/lvdb_cassiopeia_3.json) | dwarf spheroidal | 0.8017 | 2.615 | -0.001239524 | 24.104 | 0.700 | 91.00 |
| [Andromeda XIX](../data/galaxy_sources/lvdb_andromeda_19.json) | dwarf spheroidal | 0.8128 | 2.651 | -0.0003635849 | 72.360 | 0.420 | 34.00 |
| [Andromeda XIII](../data/galaxy_sources/lvdb_andromeda_13.json) | dwarf spheroidal | 0.8204 | 2.676 | -0.0006184278 | 4.077 | 0.390 | 160.00 |
| [Andromeda VII](../data/galaxy_sources/lvdb_andromeda_07.json) | dwarf spheroidal | 0.8241 | 2.688 | -0.001024709 | 17.683 | 0.870 | 94.00 |
| [Andromeda XXVII](../data/galaxy_sources/lvdb_andromeda_27.json) | dwarf spheroidal | 0.8279 | 2.700 | -0.001799912 | 9.173 | 0.600 | 150.00 |
| [Andromeda VI](../data/galaxy_sources/lvdb_andromeda_06.json) | dwarf spheroidal | 0.8318 | 2.713 | -0.001139455 | 10.956 | 0.590 | 163.00 |
| [NGC 205](../data/galaxy_sources/lvdb_ngc_0205.json) | dwarf elliptical | 0.8356 | 2.725 | -0.0008205677 | 12.536 | 0.570 | 28.00 |
| [Tucana](../data/galaxy_sources/lvdb_tucana_1.json) | dwarf spheroidal | 0.8872 | 2.894 | 0.0006004154 | 5.606 | 0.520 | 97.00 |
| [Andromeda XVIII](../data/galaxy_sources/lvdb_andromeda_18.json) | dwarf spheroidal | 1.1803 | 3.850 | -0.001124778 | 4.688 | 0.560 | 75.12 |
| [Leo P](../data/galaxy_sources/lvdb_leo_p.json) | dwarf spheroidal | 1.6218 | 5.290 | 0.0008699352 | 6.115 | 0.480 | 155.00 |
| [KDG 64](../data/galaxy_sources/lvdb_kdg_064.json) | dwarf spheroidal | 3.7497 | 12.230 | -6.137579e-05 | 5.656 | 0.650 | 22.30 |
| [Z 186-68](../data/galaxy_sources/z_186_68.json) | spiral | 47.4335 | 153.887 | 0.01069117 | 1.025 | 0.302 | 81.00 |
| [IC 1205](../data/galaxy_sources/ic_1205.json) | spiral | 48.0478 | 155.869 | 0.01083 | 0.675 | 0.886 | 94.00 |
| [ESO 264-29](../data/galaxy_sources/eso_264_29.json) | spiral | 48.1065 | 156.058 | 0.01084325 | 0.563 | 0.281 | 5.00 |
| [LEDA 85840](../data/galaxy_sources/leda_85840.json) | spiral | 48.3288 | 156.776 | 0.01089349 | 0.437 | 0.537 | 30.00 |
| [Z 313-14](../data/galaxy_sources/z_313_14.json) | spiral | 48.6673 | 157.868 | 0.01097 | 1.035 | 0.623 | 1.00 |
| [ESO 512-16](../data/galaxy_sources/eso_512_16.json) | spiral | 49.2347 | 159.698 | 0.01109822 | 0.423 | 0.600 | 175.00 |
| [NGC 5698](../data/galaxy_sources/ngc_5698.json) | spiral | 53.7103 | 174.128 | 0.01211 | 2.261 | 0.518 | 70.00 |
| [Z 163-72](../data/galaxy_sources/z_163_72.json) | lenticular | 53.8872 | 174.698 | 0.01215 | 0.800 | 0.341 | 77.00 |
| [Z 275-17](../data/galaxy_sources/z_275_17.json) | spiral | 55.4791 | 179.827 | 0.01251 | 1.341 | 0.277 | 54.00 |
| [LEDA 85410](../data/galaxy_sources/leda_85410.json) | spiral | 55.8383 | 180.984 | 0.01259125 | 0.562 | 0.859 | 66.00 |
| [IC 602](../data/galaxy_sources/ic_602.json) | spiral | 55.9212 | 181.251 | 0.01261 | 0.633 | 0.660 | 170.00 |
| [ESO 213-3](../data/galaxy_sources/eso_213_3.json) | spiral | 56.1010 | 181.831 | 0.01265068 | 0.410 | 0.661 | 140.00 |
| [NGC 6361](../data/galaxy_sources/ngc_6361.json) | spiral | 56.1423 | 181.963 | 0.01266 | 2.602 | 0.287 | 53.00 |
| [NGC 2730](../data/galaxy_sources/ngc_2730.json) | spiral | 56.5844 | 183.387 | 0.01276 | 1.641 | 0.691 | 84.00 |
| [UGC 1853](../data/galaxy_sources/ugc_1853.json) | spiral | 58.1239 | 188.344 | 0.01310825 | 0.467 | 0.240 | 160.00 |
| [Z 5-45](../data/galaxy_sources/z_5_45.json) | barred spiral | 58.1758 | 188.511 | 0.01312 | 0.557 | 0.587 | 39.00 |
| [LEDA 86486](../data/galaxy_sources/leda_86486.json) | spiral | 58.4824 | 189.499 | 0.01318937 | 0.187 | 0.840 | 170.00 |
| [NGC 643B](../data/galaxy_sources/ngc_643b.json) | lenticular | 59.6195 | 193.159 | 0.01344664 | 0.823 | 0.400 | 115.00 |
| [Z 192-26](../data/galaxy_sources/z_192_26.json) | weakly barred spiral | 61.4021 | 198.894 | 0.01385 | 1.379 | 0.264 | 134.00 |
| [Z 389-47](../data/galaxy_sources/z_389_47.json) | spiral | 61.4462 | 199.036 | 0.01386 | 1.773 | 0.379 | 148.00 |
| [Z 95-43](../data/galaxy_sources/z_95_43.json) | spiral | 62.0207 | 200.884 | 0.01399 | 1.479 | 0.204 | 121.00 |
| [Z 91-97](../data/galaxy_sources/z_91_97.json) | spiral | 62.4625 | 202.305 | 0.01409 | 1.498 | 0.315 | 17.00 |
| [Z 209-32](../data/galaxy_sources/z_209_32.json) | spiral | 62.5950 | 202.731 | 0.01412 | 0.919 | 0.726 | 173.00 |
| [Z 242-76](../data/galaxy_sources/z_242_76.json) | spiral | 62.7718 | 203.300 | 0.01416 | 1.064 | 0.331 | 15.00 |
| [Z 148-44](../data/galaxy_sources/z_148_44.json) | spiral | 62.9043 | 203.726 | 0.01419 | 1.608 | 0.599 | 179.00 |
| [Z 522-3](../data/galaxy_sources/z_522_3.json) | spiral | 63.3845 | 205.270 | 0.0142987 | 0.403 | 0.700 | 80.00 |
| [Z 137-74](../data/galaxy_sources/z_137_74.json) | spiral | 64.4063 | 208.555 | 0.01453 | 1.549 | 0.269 | 95.00 |
| [LEDA 74076](../data/galaxy_sources/leda_74076.json) | spiral | 64.5059 | 208.876 | 0.01455254 | 0.287 | 0.620 | 0.00 |
| [ESO 103-1](../data/galaxy_sources/eso_103_1.json) | spiral | 64.6823 | 209.443 | 0.01459248 | 0.347 | 0.499 | 25.00 |
| [Z 45-130](../data/galaxy_sources/z_45_130.json) | spiral | 64.7155 | 209.549 | 0.0146 | 0.343 | 0.341 | 210.00 |
| [Z 46-29](../data/galaxy_sources/z_46_29.json) | spiral | 64.8038 | 209.833 | 0.01462 | 1.406 | 0.201 | 69.00 |
| [Z 154-28](../data/galaxy_sources/z_154_28.json) | spiral | 65.2455 | 211.253 | 0.01472 | 0.607 | 0.902 | 113.00 |
| [LEDA 28784](../data/galaxy_sources/leda_28784.json) | elliptical | 66.0012 | 213.682 | 0.01489109 | 0.271 | 0.670 | 54.00 |
| [NGC 4496B](../data/galaxy_sources/ngc_4496b.json) | spiral | 66.9680 | 216.789 | 0.01511 | 1.290 | 0.634 | 111.00 |
| [NGC 4358](../data/galaxy_sources/ngc_4358.json) | elliptical | 67.1004 | 217.214 | 0.01514 | 0.273 | 0.722 | 40.00 |
| [Z 194-6](../data/galaxy_sources/z_194_6.json) | spiral | 67.3771 | 218.103 | 0.01520265 | 0.893 | 0.230 | 136.00 |
| [Z 458-6](../data/galaxy_sources/z_458_6.json) | spiral | 67.6463 | 218.968 | 0.01526362 | 1.970 | 0.251 | 20.00 |
| [LEDA 86805](../data/galaxy_sources/leda_86805.json) | spiral | 67.6613 | 219.016 | 0.01526701 | 0.337 | 0.418 | 35.00 |
| [LEDA 10275](../data/galaxy_sources/leda_10275.json) | spiral | 67.7660 | 219.352 | 0.01529072 | 0.237 | 0.599 | 50.00 |
| [Z 135-27](../data/galaxy_sources/z_135_27.json) | spiral | 70.3677 | 227.708 | 0.01588 | 1.281 | 0.341 | 18.00 |
| [LEDA 74070](../data/galaxy_sources/leda_74070.json) | spiral | 70.5187 | 228.193 | 0.0159142 | 0.603 | 0.511 | 10.00 |
| [Z 239-20](../data/galaxy_sources/z_239_20.json) | spiral | 71.2947 | 230.684 | 0.01609 | 1.527 | 0.251 | 19.00 |
| [UGC 1730](../data/galaxy_sources/ugc_1730.json) | barred spiral | 71.7607 | 232.179 | 0.01619558 | 0.267 | 0.498 | 90.00 |
| [Z 78-53](../data/galaxy_sources/z_78_53.json) | spiral | 71.8243 | 232.384 | 0.01621 | 0.900 | 0.363 | 106.00 |
| [IC 1196](../data/galaxy_sources/ic_1196.json) | spiral | 72.2657 | 233.800 | 0.01631 | 1.093 | 0.553 | 6.00 |
| [LEDA 92918](../data/galaxy_sources/leda_92918.json) | spiral | 72.4192 | 234.293 | 0.01634477 | 0.714 | 0.207 | 56.00 |
| [UGC 11310](../data/galaxy_sources/ugc_11310.json) | spiral | 72.9280 | 235.926 | 0.01646008 | 0.303 | 0.680 | 150.00 |
| [NGC 48](../data/galaxy_sources/ngc_48.json) | barred spiral | 74.2586 | 240.194 | 0.0167616 | 0.870 | 0.500 | 15.00 |
| [Z 291-71](../data/galaxy_sources/z_291_71.json) | spiral | 75.5752 | 244.417 | 0.01706 | 1.196 | 0.551 | 30.00 |
| [Z 498-70](../data/galaxy_sources/z_498_70.json) | spiral | 75.8067 | 245.160 | 0.01711247 | 0.470 | 0.657 | 153.00 |
| [Z 521-68](../data/galaxy_sources/z_521_68.json) | barred spiral | 75.8921 | 245.433 | 0.01713181 | 0.513 | 0.821 | 155.00 |
| [NGC 5648](../data/galaxy_sources/ngc_5648.json) | spiral | 76.0164 | 245.832 | 0.01716 | 0.978 | 0.698 | 176.00 |
| [NGC 5918](../data/galaxy_sources/ngc_5918.json) | spiral | 76.0605 | 245.973 | 0.01717 | 2.772 | 0.360 | 86.00 |
| [LEDA 73262](../data/galaxy_sources/leda_73262.json) | spiral | 76.2514 | 246.586 | 0.01721327 | 0.599 | 0.479 | 40.00 |
| [Z 498-34](../data/galaxy_sources/z_498_34.json) | spiral | 76.3113 | 246.778 | 0.01722684 | 0.827 | 0.239 | 105.00 |
| [Z 523-74](../data/galaxy_sources/z_523_74.json) | spiral | 76.5209 | 247.450 | 0.01727436 | 0.670 | 0.800 | 150.00 |
| [Z 211-33](../data/galaxy_sources/z_211_33.json) | spiral | 76.8105 | 248.378 | 0.01734 | 1.089 | 0.316 | 26.00 |
| [IC 2219](../data/galaxy_sources/ic_2219.json) | spiral | 77.1193 | 249.368 | 0.01741 | 1.557 | 0.503 | 176.00 |
| [NGC 105](../data/galaxy_sources/ngc_105.json) | spiral | 77.1199 | 249.370 | 0.01741014 | 0.988 | 0.779 | 167.00 |
| [Z 43-24](../data/galaxy_sources/z_43_24.json) | spiral | 77.3398 | 250.075 | 0.01746 | 0.667 | 0.822 | 123.00 |
| [Z 178-15](../data/galaxy_sources/z_178_15.json) | spiral | 77.3839 | 250.216 | 0.01747 | 0.948 | 0.776 | 137.00 |
| [NGC 2611](../data/galaxy_sources/ngc_2611.json) | spiral | 77.4722 | 250.499 | 0.01749 | 0.831 | 0.356 | 50.00 |
| [Z 134-23](../data/galaxy_sources/z_134_23.json) | spiral | 77.8250 | 251.630 | 0.01757 | 0.919 | 0.524 | 92.00 |
| [Z 134-26](../data/galaxy_sources/z_134_26.json) | spiral | 78.0897 | 252.479 | 0.01763 | 1.001 | 0.242 | 38.00 |
| [LEDA 7167](../data/galaxy_sources/leda_7167.json) | spiral | 78.1532 | 252.682 | 0.0176444 | 0.333 | 0.682 | 135.00 |
| [Z 148-108](../data/galaxy_sources/z_148_108.json) | lenticular | 78.2220 | 252.903 | 0.01766 | 1.006 | 0.324 | 89.00 |
| [Z 497-10](../data/galaxy_sources/z_497_10.json) | spiral | 78.2880 | 253.114 | 0.01767496 | 0.764 | 0.654 | 180.00 |
| [IC 2481](../data/galaxy_sources/ic_2481.json) | spiral | 78.7513 | 254.599 | 0.01778 | 0.910 | 0.598 | 161.00 |
| [Z 210-41](../data/galaxy_sources/z_210_41.json) | lenticular | 78.7954 | 254.740 | 0.01779 | 0.582 | 0.778 | 140.00 |
| [NGC 2773](../data/galaxy_sources/ngc_2773.json) | spiral | 78.9718 | 255.305 | 0.01783 | 0.868 | 0.502 | 85.00 |
| [Z 119-95](../data/galaxy_sources/z_119_95.json) | spiral | 79.7656 | 257.849 | 0.01801 | 1.008 | 0.200 | 132.00 |
| [NGC 5276](../data/galaxy_sources/ngc_5276.json) | weakly barred spiral | 80.2507 | 259.403 | 0.01812 | 0.872 | 0.511 | 154.00 |
| [Z 249-11](../data/galaxy_sources/z_249_11.json) | lenticular | 80.6034 | 260.533 | 0.0182 | 0.887 | 0.577 | 143.00 |
| [Z 35-6](../data/galaxy_sources/z_35_6.json) | barred spiral | 80.9562 | 261.663 | 0.01828 | 1.545 | 0.405 | 61.00 |
| [UGC 172](../data/galaxy_sources/ugc_172.json) | spiral | 83.3667 | 269.382 | 0.01882676 | 0.846 | 0.203 | 110.00 |
| [Z 484-20](../data/galaxy_sources/z_484_20.json) | spiral | 84.1460 | 271.876 | 0.01900356 | 1.137 | 0.280 | 65.00 |
| [Z 74-119](../data/galaxy_sources/z_74_119.json) | spiral | 84.3067 | 272.391 | 0.01904 | 0.932 | 0.747 | 70.00 |
| [NGC 5265](../data/galaxy_sources/ngc_5265.json) | spiral | 84.5271 | 273.096 | 0.01909 | 0.855 | 0.815 | 88.00 |
| [LEDA 37778](../data/galaxy_sources/leda_37778.json) | lenticular | 85.4526 | 276.058 | 0.0193 | 0.463 | 0.641 | 76.00 |
| [Z 521-74](../data/galaxy_sources/z_521_74.json) | spiral | 85.6749 | 276.769 | 0.01935044 | 0.530 | 0.321 | 85.00 |
| [Z 74-146](../data/galaxy_sources/z_74_146.json) | spiral | 86.5984 | 279.724 | 0.01956 | 0.903 | 0.693 | 81.00 |
| [Z 124-62](../data/galaxy_sources/z_124_62.json) | barred spiral | 86.7747 | 280.287 | 0.0196 | 0.534 | 0.749 | 18.00 |
| [IC 2214](../data/galaxy_sources/ic_2214.json) | spiral | 87.3034 | 281.979 | 0.01972 | 0.856 | 0.923 | 144.00 |
| [Z 44-69](../data/galaxy_sources/z_44_69.json) | spiral | 87.4356 | 282.402 | 0.01975 | 0.635 | 0.748 | 27.00 |
| [NGC 3719](../data/galaxy_sources/ngc_3719.json) | spiral | 87.6374 | 283.047 | 0.01979579 | 1.595 | 0.696 | 18.00 |
| [Z 251-17](../data/galaxy_sources/z_251_17.json) | spiral | 87.6559 | 283.106 | 0.0198 | 0.790 | 0.420 | 77.00 |
| [Z 262-27](../data/galaxy_sources/z_262_27.json) | spiral | 87.9643 | 284.093 | 0.01987 | 0.985 | 0.269 | 71.00 |
| [LEDA 86552](../data/galaxy_sources/leda_86552.json) | spiral | 88.5371 | 285.924 | 0.02 | 0.613 | 0.274 | 117.00 |
| [Z 7-35](../data/galaxy_sources/z_7_35.json) | spiral | 88.8895 | 287.051 | 0.02008 | 1.004 | 0.329 | 118.00 |
| [NGC 3849](../data/galaxy_sources/ngc_3849.json) | spiral | 89.1098 | 287.755 | 0.02013 | 0.943 | 0.604 | 29.00 |
| [Z 122-31](../data/galaxy_sources/z_122_31.json) | spiral | 89.2420 | 288.178 | 0.02016 | 0.746 | 0.677 | 4.00 |
| [Z 431-73](../data/galaxy_sources/z_431_73.json) | spiral | 89.3034 | 288.374 | 0.02017394 | 0.619 | 0.864 | 49.00 |
| [IC 2572](../data/galaxy_sources/ic_2572.json) | spiral | 90.5194 | 292.262 | 0.02045 | 0.821 | 0.411 | 26.00 |
| [MCG+08-28-036](../data/galaxy_sources/mcg_08_28_036.json) | barred spiral | 90.8278 | 293.247 | 0.02052 | 0.760 | 0.608 | 33.00 |
| [NGC 2711](../data/galaxy_sources/ngc_2711.json) | barred spiral | 90.9159 | 293.528 | 0.02054 | 0.651 | 0.778 | 83.00 |
| [Z 241-85](../data/galaxy_sources/z_241_85.json) | spiral | 91.2242 | 294.514 | 0.02061 | 0.669 | 0.961 | 140.00 |
| [Z 215-7](../data/galaxy_sources/z_215_7.json) | weakly barred spiral | 91.3123 | 294.795 | 0.02063 | 1.445 | 0.466 | 29.00 |
| [UGC 5392](../data/galaxy_sources/ugc_5392.json) | spiral | 91.4884 | 295.358 | 0.02067 | 1.516 | 0.138 | 10.00 |
| [LEDA 12538](../data/galaxy_sources/leda_12538.json) | lenticular | 92.0169 | 297.047 | 0.02079 | 0.307 | 0.917 | 68.00 |
| [LEDA 85390](../data/galaxy_sources/leda_85390.json) | spiral | 92.7084 | 299.256 | 0.02094701 | 0.478 | 0.699 | 72.00 |
| [NGC 6941](../data/galaxy_sources/ngc_6941.json) | spiral | 92.9634 | 300.071 | 0.02100493 | 1.807 | 0.702 | 128.00 |
| [Z 121-85](../data/galaxy_sources/z_121_85.json) | spiral | 93.2059 | 300.845 | 0.02106 | 1.136 | 0.333 | 128.00 |
| [Z 267-44](../data/galaxy_sources/z_267_44.json) | spiral | 93.6463 | 302.252 | 0.02116 | 1.467 | 0.363 | 6.00 |
| [MCG+01-60-027](../data/galaxy_sources/mcg_01_60_027.json) | spiral | 93.8487 | 302.899 | 0.02120598 | 1.156 | 0.199 | 25.00 |
| [Z 207-53](../data/galaxy_sources/z_207_53.json) | spiral | 93.9104 | 303.096 | 0.02122 | 1.788 | 0.524 | 13.00 |
| [Z 403-13](../data/galaxy_sources/z_403_13.json) | spiral | 93.9237 | 303.138 | 0.02122302 | 0.931 | 0.257 | 172.00 |
| [LEDA 92868](../data/galaxy_sources/leda_92868.json) | spiral | 94.3507 | 304.502 | 0.02132 | 1.717 | 0.162 | 41.00 |
| [Z 267-39](../data/galaxy_sources/z_267_39.json) | spiral | 94.5709 | 305.205 | 0.02137 | 1.117 | 0.366 | 179.00 |
| [NGC 5019](../data/galaxy_sources/ngc_5019.json) | spiral | 94.8791 | 306.189 | 0.02144 | 0.844 | 0.867 | 114.00 |
| [Z 179-7](../data/galaxy_sources/z_179_7.json) | spiral | 95.3633 | 307.735 | 0.02155 | 1.149 | 0.249 | 51.00 |
| [Z 100-22](../data/galaxy_sources/z_100_22.json) | spiral | 95.5835 | 308.438 | 0.0216 | 0.798 | 0.799 | 153.00 |
| [LEDA 84280](../data/galaxy_sources/leda_84280.json) | spiral | 96.7720 | 312.232 | 0.02187 | 0.599 | 0.730 | 98.00 |
| [Z 167-26](../data/galaxy_sources/z_167_26.json) | lenticular | 96.9040 | 312.653 | 0.0219 | 1.507 | 0.673 | 142.00 |
| [Z 157-46](../data/galaxy_sources/z_157_46.json) | spiral | 97.2122 | 313.637 | 0.02197 | 1.084 | 0.345 | 100.00 |
| [UGC 12145](../data/galaxy_sources/ugc_12145.json) | spiral | 97.2256 | 313.680 | 0.02197305 | 0.759 | 0.232 | 85.00 |
| [Z 94-99](../data/galaxy_sources/z_94_99.json) | spiral | 97.2562 | 313.777 | 0.02198 | 1.161 | 0.201 | 172.00 |
| [LEDA 85406](../data/galaxy_sources/leda_85406.json) | spiral | 97.5859 | 314.829 | 0.02205491 | 0.553 | 0.416 | 158.00 |
| [MCG-02-01-042](../data/galaxy_sources/mcg_02_01_042.json) | lenticular | 98.1804 | 316.727 | 0.02219 | 0.731 | 0.441 | 2.00 |
| [Z 236-12](../data/galaxy_sources/z_236_12.json) | spiral | 98.3565 | 317.288 | 0.02223 | 0.579 | 0.892 | 168.00 |
| [Z 235-23](../data/galaxy_sources/z_235_23.json) | spiral | 98.4005 | 317.429 | 0.02224 | 0.671 | 0.756 | 150.00 |
| [UGC 5802](../data/galaxy_sources/ugc_5802.json) | spiral | 98.4005 | 317.429 | 0.02224 | 1.077 | 0.155 | 2.00 |
| [LEDA 3091198](../data/galaxy_sources/leda_3091198.json) | lenticular | 98.4445 | 317.569 | 0.02225 | 0.487 | 0.427 | 80.00 |
| [LEDA 44656](../data/galaxy_sources/leda_44656.json) | elliptical | 98.4445 | 317.569 | 0.02225 | 0.390 | 0.962 | 38.00 |
| [LEDA 1073319](../data/galaxy_sources/leda_1073319.json) | lenticular | 98.5325 | 317.850 | 0.02227 | 0.592 | 0.513 | 135.00 |
| [MCG-01-07-022](../data/galaxy_sources/mcg_01_07_022.json) | spiral | 98.5392 | 317.871 | 0.02227152 | 0.893 | 0.952 | 32.00 |
| [Z 182-8](../data/galaxy_sources/z_182_8.json) | spiral | 98.6205 | 318.131 | 0.02229 | 0.908 | 0.411 | 25.00 |
| [Z 315-2](../data/galaxy_sources/z_315_2.json) | spiral | 98.6645 | 318.271 | 0.0223 | 0.525 | 0.947 | 106.00 |
| [Z 192-5](../data/galaxy_sources/z_192_5.json) | spiral | 98.6645 | 318.271 | 0.0223 | 0.531 | 0.737 | 132.00 |
| [Z 77-47](../data/galaxy_sources/z_77_47.json) | spiral | 98.6645 | 318.271 | 0.0223 | 0.554 | 0.604 | 29.00 |
| [LEDA 1112286](../data/galaxy_sources/leda_1112286.json) | spiral | 98.6645 | 318.271 | 0.0223 | 0.492 | 0.385 | 118.00 |
| [Z 170-23](../data/galaxy_sources/z_170_23.json) | spiral | 98.7085 | 318.412 | 0.02231 | 0.656 | 0.628 | 82.00 |
| [LEDA 1189082](../data/galaxy_sources/leda_1189082.json) | spiral | 98.7525 | 318.552 | 0.02232 | 0.426 | 0.344 | 82.00 |
| [Z 17-43](../data/galaxy_sources/z_17_43.json) | barred spiral | 98.7965 | 318.692 | 0.02233 | 0.408 | 0.766 | 68.00 |
| [UGC 9272](../data/galaxy_sources/ugc_9272.json) | spiral | 98.8406 | 318.833 | 0.02234 | 0.849 | 0.390 | 36.00 |
| [LEDA 27980](../data/galaxy_sources/leda_27980.json) | spiral | 98.8406 | 318.833 | 0.02234 | 0.347 | 0.858 | 88.00 |
| [Z 183-15](../data/galaxy_sources/z_183_15.json) | lenticular | 98.8846 | 318.973 | 0.02235 | 0.721 | 0.656 | 15.00 |
| [Z 262-53](../data/galaxy_sources/z_262_53.json) | spiral | 98.9286 | 319.114 | 0.02236 | 0.678 | 0.530 | 1.00 |
| [LEDA 29395](../data/galaxy_sources/leda_29395.json) | spiral | 98.9286 | 319.114 | 0.02236 | 0.386 | 0.805 | 162.00 |
| [LEDA 31969](../data/galaxy_sources/leda_31969.json) | spiral | 98.9286 | 319.114 | 0.02236 | 0.741 | 0.448 | 13.00 |
| [IC 3847](../data/galaxy_sources/ic_3847.json) | spiral | 98.9286 | 319.114 | 0.02236 | 0.344 | 0.764 | 127.00 |
| [Z 241-33](../data/galaxy_sources/z_241_33.json) | elliptical | 98.9726 | 319.254 | 0.02237 | 0.605 | 0.820 | 54.00 |
| [UGC 7295](../data/galaxy_sources/ugc_7295.json) | spiral | 98.9726 | 319.254 | 0.02237 | 0.849 | 0.367 | 180.00 |
| [SDSS J102750.53+020708.6](../data/galaxy_sources/sdss_j102750_53_020708_6.json) | spiral | 98.9726 | 319.254 | 0.02237 | 0.210 | 0.781 | 190.00 |
| [Z 437-28](../data/galaxy_sources/z_437_28.json) | spiral | 99.0422 | 319.476 | 0.02238581 | 1.051 | 0.394 | 14.00 |
| [Z 119-14](../data/galaxy_sources/z_119_14.json) | spiral | 99.0606 | 319.535 | 0.02239 | 0.519 | 0.613 | 69.00 |
| [MCG+06-28-016](../data/galaxy_sources/mcg_06_28_016.json) | spiral | 99.1046 | 319.675 | 0.0224 | 0.486 | 0.615 | 124.00 |
| [LEDA 82527](../data/galaxy_sources/leda_82527.json) | spiral | 99.1046 | 319.675 | 0.0224 | 0.522 | 0.330 | 35.00 |
| [Z 72-56](../data/galaxy_sources/z_72_56.json) | barred spiral | 99.1486 | 319.816 | 0.02241 | 0.733 | 0.704 | 31.00 |
| [LEDA 38540](../data/galaxy_sources/leda_38540.json) | spiral | 99.1926 | 319.956 | 0.02242 | 0.546 | 0.739 | 79.00 |
| [LEDA 1217343](../data/galaxy_sources/leda_1217343.json) | spiral | 99.1926 | 319.956 | 0.02242 | 0.646 | 0.252 | 96.00 |
| [Z 39-66](../data/galaxy_sources/z_39_66.json) | spiral | 99.2806 | 320.237 | 0.02244 | 0.669 | 0.577 | 174.00 |
| [Z 151-72](../data/galaxy_sources/z_151_72.json) | spiral | 99.3686 | 320.517 | 0.02246 | 0.622 | 0.923 | 35.00 |
| [LEDA 39020](../data/galaxy_sources/leda_39020.json) | spiral | 99.4566 | 320.798 | 0.02248 | 0.509 | 0.555 | 64.00 |
| [Z 313-29](../data/galaxy_sources/z_313_29.json) | spiral | 99.5006 | 320.939 | 0.02249 | 0.761 | 0.419 | 45.00 |
| [LEDA 9694](../data/galaxy_sources/leda_9694.json) | spiral | 99.5006 | 320.939 | 0.02249 | 0.493 | 0.535 | 97.00 |
| [LEDA 1180028](../data/galaxy_sources/leda_1180028.json) | spiral | 99.5886 | 321.219 | 0.02251 | 0.490 | 0.517 | 172.00 |
| [ESO 266-29](../data/galaxy_sources/eso_266_29.json) | spiral | 99.6203 | 321.320 | 0.02251718 | 0.267 | 0.719 | 140.00 |
| [Z 241-32](../data/galaxy_sources/z_241_32.json) | lenticular | 99.6326 | 321.360 | 0.02252 | 0.661 | 0.435 | 87.00 |
| [Z 118-61](../data/galaxy_sources/z_118_61.json) | spiral | 99.6326 | 321.360 | 0.02252 | 1.420 | 0.381 | 170.00 |
| [LEDA 27044](../data/galaxy_sources/leda_27044.json) | spiral | 99.6326 | 321.360 | 0.02252 | 0.636 | 0.335 | 165.00 |
| [Z 35-16](../data/galaxy_sources/z_35_16.json) | spiral | 99.6326 | 321.360 | 0.02252 | 0.748 | 0.359 | 58.00 |
| [UGC 12787](../data/galaxy_sources/ugc_12787.json) | spiral | 99.6428 | 321.392 | 0.0225223 | 0.886 | 0.318 | 1.00 |
| [UGC 11930](../data/galaxy_sources/ugc_11930.json) | spiral | 99.6578 | 321.440 | 0.02252572 | 1.428 | 0.280 | 60.00 |
| [LEDA 38719](../data/galaxy_sources/leda_38719.json) | spiral | 99.6767 | 321.500 | 0.02253 | 0.432 | 0.755 | 79.00 |
| [Z 263-5](../data/galaxy_sources/z_263_5.json) | spiral | 99.7207 | 321.640 | 0.02254 | 0.992 | 0.526 | 113.00 |
| [Z 126-18](../data/galaxy_sources/z_126_18.json) | spiral | 99.7207 | 321.640 | 0.02254 | 0.689 | 0.937 | 33.00 |
| [LEDA 48606](../data/galaxy_sources/leda_48606.json) | barred lenticular | 99.7207 | 321.640 | 0.02254 | 0.554 | 0.742 | 122.00 |
| [MCG+06-21-034](../data/galaxy_sources/mcg_06_21_034.json) | spiral | 99.7647 | 321.781 | 0.02255 | 0.487 | 0.903 | 61.00 |
| [Z 40-58](../data/galaxy_sources/z_40_58.json) | spiral | 99.7647 | 321.781 | 0.02255 | 0.434 | 0.881 | 25.00 |
| [Z 433-28](../data/galaxy_sources/z_433_28.json) | spiral | 99.7779 | 321.823 | 0.02255302 | 1.412 | 0.323 | 49.00 |
| [Z 12-24](../data/galaxy_sources/z_12_24.json) | weakly barred spiral | 99.8087 | 321.921 | 0.02256 | 0.699 | 0.719 | 74.00 |
| [LEDA 89713](../data/galaxy_sources/leda_89713.json) | spiral | 99.8087 | 321.921 | 0.02256 | 0.468 | 0.849 | 48.00 |
| [Z 16-75](../data/galaxy_sources/z_16_75.json) | barred spiral | 99.8087 | 321.921 | 0.02256 | 0.682 | 0.793 | 167.00 |
| [Z 190-19](../data/galaxy_sources/z_190_19.json) | spiral | 99.8087 | 321.921 | 0.02256 | 2.116 | 0.291 | 159.00 |
| [NGC 7451](../data/galaxy_sources/ngc_7451.json) | spiral | 99.8230 | 321.967 | 0.02256325 | 1.264 | 0.460 | 62.00 |
| [LEDA 38655](../data/galaxy_sources/leda_38655.json) | lenticular | 99.8527 | 322.061 | 0.02257 | 0.324 | 0.930 | 27.00 |
| [LEDA 38909](../data/galaxy_sources/leda_38909.json) | spiral | 99.8527 | 322.061 | 0.02257 | 0.367 | 0.846 | 54.00 |
| [Z 40-11](../data/galaxy_sources/z_40_11.json) | barred spiral | 99.8967 | 322.202 | 0.02258 | 0.573 | 0.908 | 38.00 |
| [LEDA 38133](../data/galaxy_sources/leda_38133.json) | spiral | 99.8967 | 322.202 | 0.02258 | 0.417 | 0.826 | 167.00 |
| [Z 15-47](../data/galaxy_sources/z_15_47.json) | spiral | 99.8967 | 322.202 | 0.02258 | 0.581 | 0.342 | 23.00 |
| [IC 2967](../data/galaxy_sources/ic_2967.json) | elliptical | 99.9407 | 322.342 | 0.02259 | 0.893 | 0.807 | 21.00 |
| [MCG+06-20-035](../data/galaxy_sources/mcg_06_20_035.json) | spiral | 99.9847 | 322.482 | 0.0226 | 0.434 | 0.727 | 3.00 |
| [NGC 301](../data/galaxy_sources/ngc_301.json) | spiral | 99.9847 | 322.482 | 0.0226 | 0.669 | 0.822 | 72.00 |
| [Z 128-90](../data/galaxy_sources/z_128_90.json) | spiral | 99.9847 | 322.482 | 0.0226 | 0.874 | 0.678 | 10.00 |
| [Z 135-52](../data/galaxy_sources/z_135_52.json) | spiral | 100.0332 | 322.637 | 0.02261103 | 1.029 | 0.766 | 176.00 |
| [Z 388-12](../data/galaxy_sources/z_388_12.json) | spiral | 100.0482 | 322.685 | 0.02261444 | 0.714 | 0.897 | 139.00 |
| [Z 389-50](../data/galaxy_sources/z_389_50.json) | spiral | 100.0632 | 322.733 | 0.02261786 | 0.571 | 0.612 | 109.00 |
| [LEDA 86951](../data/galaxy_sources/leda_86951.json) | lenticular | 100.0727 | 322.763 | 0.02262 | 0.436 | 0.631 | 130.00 |
| [Z 39-17](../data/galaxy_sources/z_39_17.json) | barred spiral | 100.0727 | 322.763 | 0.02262 | 0.507 | 0.806 | 120.00 |
| [Z 101-22](../data/galaxy_sources/z_101_22.json) | spiral | 100.1562 | 323.030 | 0.022639 | 0.495 | 0.540 | 162.00 |
| [MCG+11-12-018](../data/galaxy_sources/mcg_11_12_018.json) | spiral | 100.1607 | 323.044 | 0.02264 | 0.581 | 0.852 | 164.00 |
| [Z 65-36](../data/galaxy_sources/z_65_36.json) | lenticular | 100.1607 | 323.044 | 0.02264 | 0.824 | 0.536 | 35.00 |
| [LEDA 38590](../data/galaxy_sources/leda_38590.json) | spiral | 100.1607 | 323.044 | 0.02264 | 0.366 | 0.759 | 69.00 |
| [IC 3976](../data/galaxy_sources/ic_3976.json) | lenticular | 100.1607 | 323.044 | 0.02264 | 0.607 | 0.488 | 154.00 |
| [Z 455-18](../data/galaxy_sources/z_455_18.json) | barred spiral | 100.1684 | 323.068 | 0.02264175 | 1.324 | 0.173 | 120.00 |
| [Z 480-41](../data/galaxy_sources/z_480_41.json) | spiral | 100.1834 | 323.116 | 0.02264516 | 1.133 | 0.197 | 10.00 |
| [LEDA 38871](../data/galaxy_sources/leda_38871.json) | spiral | 100.2047 | 323.184 | 0.02265 | 0.458 | 0.742 | 133.00 |
| [Z 14-38](../data/galaxy_sources/z_14_38.json) | spiral | 100.2047 | 323.184 | 0.02265 | 0.664 | 0.461 | 91.00 |
| [MCG+05-41-001](../data/galaxy_sources/mcg_05_41_001.json) | spiral | 100.2047 | 323.184 | 0.02265 | 0.596 | 0.845 | 54.00 |
| [MCG+05-28-050](../data/galaxy_sources/mcg_05_28_050.json) | spiral | 100.2927 | 323.465 | 0.02267 | 1.137 | 0.195 | 41.00 |
| [LEDA 89720](../data/galaxy_sources/leda_89720.json) | spiral | 100.2927 | 323.465 | 0.02267 | 0.584 | 0.250 | 144.00 |
| [UGC 8176](../data/galaxy_sources/ugc_8176.json) | spiral | 100.3807 | 323.745 | 0.02269 | 1.398 | 0.133 | 99.00 |
| [Z 389-41](../data/galaxy_sources/z_389_41.json) | spiral | 100.4537 | 323.978 | 0.02270659 | 0.915 | 0.296 | 18.00 |
| [LEDA 27634](../data/galaxy_sources/leda_27634.json) | spiral | 100.4687 | 324.026 | 0.02271 | 1.040 | 0.169 | 14.00 |
| [Z 40-64](../data/galaxy_sources/z_40_64.json) | spiral | 100.4687 | 324.026 | 0.02271 | 0.790 | 0.660 | 26.00 |
| [LEDA 35398](../data/galaxy_sources/leda_35398.json) | spiral | 100.5127 | 324.166 | 0.02272 | 0.591 | 0.658 | 173.00 |
| [Z 65-5](../data/galaxy_sources/z_65_5.json) | spiral | 100.6007 | 324.447 | 0.02274 | 0.549 | 0.850 | 58.00 |
| [LEDA 44636](../data/galaxy_sources/leda_44636.json) | lenticular | 100.6007 | 324.447 | 0.02274 | 0.273 | 0.700 | 40.00 |
| [Z 116-20](../data/galaxy_sources/z_116_20.json) | spiral | 100.6112 | 324.481 | 0.0227424 | 0.667 | 0.559 | 110.00 |
| [Z 382-1](../data/galaxy_sources/z_382_1.json) | spiral | 100.6447 | 324.587 | 0.02275 | 0.820 | 0.831 | 24.00 |
| [Z 102-8](../data/galaxy_sources/z_102_8.json) | barred spiral | 100.6447 | 324.587 | 0.02275 | 0.633 | 0.815 | 113.00 |
| [Z 495-4](../data/galaxy_sources/z_495_4.json) | spiral | 100.6789 | 324.697 | 0.02275779 | 0.622 | 0.822 | 54.00 |
| [Z 313-19](../data/galaxy_sources/z_313_19.json) | spiral | 100.6887 | 324.728 | 0.02276 | 0.983 | 0.373 | 112.00 |
| [NGC 7505](../data/galaxy_sources/ngc_7505.json) | lenticular | 100.6887 | 324.728 | 0.02276 | 0.765 | 0.360 | 109.00 |
| [Z 11-94](../data/galaxy_sources/z_11_94.json) | weakly barred spiral | 100.6887 | 324.728 | 0.02276 | 1.122 | 0.626 | 43.00 |
| [LEDA 30951](../data/galaxy_sources/leda_30951.json) | spiral | 100.7327 | 324.868 | 0.02277 | 0.298 | 0.669 | 82.00 |
| [Z 73-59](../data/galaxy_sources/z_73_59.json) | spiral | 100.7327 | 324.868 | 0.02277 | 0.958 | 0.399 | 154.00 |
| [Z 455-39](../data/galaxy_sources/z_455_39.json) | spiral | 100.7540 | 324.936 | 0.02277486 | 1.458 | 0.238 | 106.00 |
| [LEDA 37272](../data/galaxy_sources/leda_37272.json) | spiral | 100.7767 | 325.008 | 0.02278 | 0.664 | 0.207 | 145.00 |
| [2MASX J13223734-0230502](../data/galaxy_sources/galaxy_2masx_j13223734_0230502.json) | spiral | 100.7767 | 325.008 | 0.02278 | 0.223 | 0.740 | 50.00 |
| [Z 74-19](../data/galaxy_sources/z_74_19.json) | spiral | 100.7767 | 325.008 | 0.02278 | 0.551 | 0.636 | 81.00 |
| [LEDA 85641](../data/galaxy_sources/leda_85641.json) | spiral | 100.8141 | 325.128 | 0.02278851 | 0.491 | 0.367 | 96.00 |
| [LEDA 31461](../data/galaxy_sources/leda_31461.json) | spiral | 100.8207 | 325.149 | 0.02279 | 0.439 | 0.445 | 75.00 |
| [LEDA 37590](../data/galaxy_sources/leda_37590.json) | spiral | 100.9086 | 325.429 | 0.02281 | 0.402 | 0.765 | 42.00 |
| [LEDA 89691](../data/galaxy_sources/leda_89691.json) | spiral | 100.9086 | 325.429 | 0.02281 | 0.573 | 0.564 | 85.00 |
| [Z 14-88](../data/galaxy_sources/z_14_88.json) | lenticular | 100.9086 | 325.429 | 0.02281 | 0.834 | 0.410 | 148.00 |
| [ESO 384-1](../data/galaxy_sources/eso_384_1.json) | spiral | 100.9192 | 325.463 | 0.02281241 | 0.580 | 0.640 | 100.00 |
| [IC 2385](../data/galaxy_sources/ic_2385.json) | spiral | 100.9966 | 325.710 | 0.02283 | 0.796 | 0.695 | 18.00 |
| [LEDA 35343](../data/galaxy_sources/leda_35343.json) | spiral | 100.9966 | 325.710 | 0.02283 | 0.391 | 0.434 | 178.00 |
| [Z 50-56](../data/galaxy_sources/z_50_56.json) | spiral | 100.9966 | 325.710 | 0.02283 | 1.033 | 0.990 | 18.00 |
| [ESO 427-11](../data/galaxy_sources/eso_427_11.json) | spiral | 101.0078 | 325.745 | 0.02283254 | 0.237 | 0.781 | 160.00 |
| [Z 37-35](../data/galaxy_sources/z_37_35.json) | spiral | 101.0406 | 325.850 | 0.02284 | 0.843 | 0.386 | 18.00 |
| [LEDA 83570](../data/galaxy_sources/leda_83570.json) | lenticular | 101.0406 | 325.850 | 0.02284 | 0.553 | 0.562 | 119.00 |
| [ESO 555-40](../data/galaxy_sources/eso_555_40.json) | spiral | 101.0799 | 325.975 | 0.02284893 | 0.700 | 0.260 | 40.00 |
| [Z 207-47](../data/galaxy_sources/z_207_47.json) | elliptical | 101.1286 | 326.131 | 0.02286 | 0.936 | 0.712 | 79.00 |
| [Z 118-9](../data/galaxy_sources/z_118_9.json) | spiral | 101.1286 | 326.131 | 0.02286 | 0.706 | 0.703 | 0.00 |
| [Z 243-49](../data/galaxy_sources/z_243_49.json) | spiral | 101.1286 | 326.131 | 0.02286 | 0.584 | 0.816 | 8.00 |
| [LEDA 1070338](../data/galaxy_sources/leda_1070338.json) | spiral | 101.1286 | 326.131 | 0.02286 | 0.553 | 0.277 | 4.00 |
| [LEDA 1189171](../data/galaxy_sources/leda_1189171.json) | spiral | 101.1286 | 326.131 | 0.02286 | 0.447 | 0.283 | 53.00 |
| [Z 98-57](../data/galaxy_sources/z_98_57.json) | spiral | 101.1745 | 326.277 | 0.02287044 | 0.490 | 0.636 | 79.00 |
| [Z 15-28](../data/galaxy_sources/z_15_28.json) | barred spiral | 101.2166 | 326.411 | 0.02288 | 0.671 | 0.800 | 8.00 |
| [UGC 3607](../data/galaxy_sources/ugc_3607.json) | barred spiral | 101.2196 | 326.421 | 0.02288068 | 0.343 | 0.802 | 20.00 |
| [MCG-02-03-053](../data/galaxy_sources/mcg_02_03_053.json) | weakly barred spiral | 101.2606 | 326.551 | 0.02289 | 0.979 | 0.273 | 43.00 |
| [Z 212-15](../data/galaxy_sources/z_212_15.json) | spiral | 101.2606 | 326.551 | 0.02289 | 0.614 | 0.457 | 86.00 |
| [Z 14-76](../data/galaxy_sources/z_14_76.json) | lenticular | 101.2606 | 326.551 | 0.02289 | 0.821 | 0.414 | 135.00 |
| [Z 39-101](../data/galaxy_sources/z_39_101.json) | spiral | 101.3046 | 326.692 | 0.0229 | 1.142 | 0.205 | 26.00 |
| [Z 455-38](../data/galaxy_sources/z_455_38.json) | spiral | 101.3398 | 326.804 | 0.02290799 | 0.638 | 0.769 | 165.00 |
| [ZW VIII 166](../data/galaxy_sources/zw_viii_166.json) | barred spiral | 101.3486 | 326.832 | 0.02291 | 0.329 | 0.809 | 138.00 |
| [Z 74-110](../data/galaxy_sources/z_74_110.json) | spiral | 101.3486 | 326.832 | 0.02291 | 1.037 | 0.240 | 119.00 |
| [MCG+00-06-031](../data/galaxy_sources/mcg_00_06_031.json) | spiral | 101.3698 | 326.900 | 0.02291481 | 1.563 | 0.127 | 111.00 |
| [Z 313-8](../data/galaxy_sources/z_313_8.json) | lenticular | 101.3926 | 326.972 | 0.02292 | 0.780 | 0.617 | 8.00 |
| [Z 236-33](../data/galaxy_sources/z_236_33.json) | spiral | 101.3926 | 326.972 | 0.02292 | 0.442 | 0.981 | 19.00 |
| [Z 148-83](../data/galaxy_sources/z_148_83.json) | spiral | 101.3926 | 326.972 | 0.02292 | 0.948 | 0.239 | 90.00 |
| [MCG+00-60-041](../data/galaxy_sources/mcg_00_60_041.json) | spiral | 101.3926 | 326.972 | 0.02292 | 1.063 | 0.242 | 20.00 |
| [Z 36-32](../data/galaxy_sources/z_36_32.json) | elliptical | 101.3926 | 326.972 | 0.02292 | 0.581 | 0.613 | 120.00 |
| [NGC 5213](../data/galaxy_sources/ngc_5213.json) | weakly barred spiral | 101.3926 | 326.972 | 0.02292 | 0.972 | 0.680 | 136.00 |
| [IC 1836](../data/galaxy_sources/ic_1836.json) | spiral | 101.3998 | 326.995 | 0.02292164 | 0.757 | 0.896 | 22.00 |
| [Z 499-114](../data/galaxy_sources/z_499_114.json) | spiral | 101.3998 | 326.995 | 0.02292164 | 0.477 | 0.568 | 71.00 |
| [Z 500-18](../data/galaxy_sources/z_500_18.json) | spiral | 101.4449 | 327.139 | 0.02293188 | 2.181 | 0.309 | 177.00 |
| [ESO 427-15](../data/galaxy_sources/eso_427_15.json) | spiral | 101.4704 | 327.220 | 0.02293769 | 0.383 | 0.621 | 170.00 |
| [Z 181-68](../data/galaxy_sources/z_181_68.json) | spiral | 101.4749 | 327.235 | 0.02293871 | 0.913 | 0.294 | 52.00 |
| [LEDA 31420](../data/galaxy_sources/leda_31420.json) | spiral | 101.4806 | 327.253 | 0.02294 | 0.566 | 0.395 | 176.00 |
| [Z 39-137](../data/galaxy_sources/z_39_137.json) | spiral | 101.4806 | 327.253 | 0.02294 | 0.788 | 0.443 | 154.00 |
| [Z 212-8](../data/galaxy_sources/z_212_8.json) | spiral | 101.5246 | 327.393 | 0.02295 | 0.973 | 0.240 | 85.00 |
| [Z 39-165](../data/galaxy_sources/z_39_165.json) | spiral | 101.5246 | 327.393 | 0.02295 | 0.829 | 0.954 | 92.00 |
| [Z 72-81](../data/galaxy_sources/z_72_81.json) | spiral | 101.5246 | 327.393 | 0.02295 | 1.079 | 0.289 | 35.00 |
| [Z 45-89](../data/galaxy_sources/z_45_89.json) | spiral | 101.5246 | 327.393 | 0.02295 | 0.606 | 0.368 | 39.00 |
| [Z 45-55](../data/galaxy_sources/z_45_55.json) | barred spiral | 101.5350 | 327.426 | 0.02295237 | 0.776 | 0.737 | 177.00 |
| [ESO 110-21](../data/galaxy_sources/eso_110_21.json) | spiral | 101.5410 | 327.446 | 0.02295373 | 0.561 | 0.929 | 53.00 |
| [Z 456-15](../data/galaxy_sources/z_456_15.json) | spiral | 101.6101 | 327.666 | 0.02296944 | 2.073 | 0.122 | 110.00 |
| [Z 211-51](../data/galaxy_sources/z_211_51.json) | barred spiral | 101.6126 | 327.674 | 0.02297 | 0.857 | 0.531 | 42.00 |
| [Z 524-42](../data/galaxy_sources/z_524_42.json) | spiral | 101.6401 | 327.762 | 0.02297627 | 0.473 | 0.440 | 25.00 |
| [Z 496-69](../data/galaxy_sources/z_496_69.json) | spiral | 101.6401 | 327.762 | 0.02297627 | 0.610 | 0.818 | 93.00 |
| [Z 501-113](../data/galaxy_sources/z_501_113.json) | spiral | 101.6702 | 327.857 | 0.02298309 | 0.675 | 0.548 | 32.00 |
| [Z 37-29](../data/galaxy_sources/z_37_29.json) | spiral | 101.7006 | 327.954 | 0.02299 | 0.829 | 0.505 | 14.00 |
| [Z 42-74](../data/galaxy_sources/z_42_74.json) | spiral | 101.7006 | 327.954 | 0.02299 | 0.695 | 0.758 | 44.00 |
| [Z 14-46](../data/galaxy_sources/z_14_46.json) | spiral | 101.7006 | 327.954 | 0.02299 | 0.503 | 0.765 | 21.00 |
| [LEDA 46838](../data/galaxy_sources/leda_46838.json) | spiral | 101.7006 | 327.954 | 0.02299 | 0.396 | 0.467 | 66.00 |
| [NGC 7430](../data/galaxy_sources/ngc_7430.json) | lenticular | 101.7152 | 328.001 | 0.02299334 | 1.115 | 0.514 | 58.00 |
| [Z 118-10](../data/galaxy_sources/z_118_10.json) | lenticular | 101.7446 | 328.094 | 0.023 | 1.001 | 0.541 | 35.00 |
| [Z 36-36](../data/galaxy_sources/z_36_36.json) | lenticular | 101.7446 | 328.094 | 0.023 | 0.661 | 0.662 | 141.00 |
| [Z 181-49](../data/galaxy_sources/z_181_49.json) | spiral | 101.7885 | 328.235 | 0.02301 | 0.692 | 0.579 | 74.00 |
| [Z 220-26](../data/galaxy_sources/z_220_26.json) | spiral | 101.8765 | 328.515 | 0.02303 | 1.102 | 0.536 | 83.00 |
| [LEDA 44652](../data/galaxy_sources/leda_44652.json) | lenticular | 101.8765 | 328.515 | 0.02303 | 0.487 | 0.505 | 17.00 |
| [LEDA 86876](../data/galaxy_sources/leda_86876.json) | barred spiral | 101.9205 | 328.656 | 0.02304 | 0.407 | 0.662 | 102.00 |
| [Z 37-41](../data/galaxy_sources/z_37_41.json) | spiral | 101.9205 | 328.656 | 0.02304 | 0.618 | 0.830 | 94.00 |
| [LEDA 37315](../data/galaxy_sources/leda_37315.json) | spiral | 101.9205 | 328.656 | 0.02304 | 0.596 | 0.282 | 107.00 |
| [Z 160-233](../data/galaxy_sources/z_160_233.json) | lenticular | 101.9205 | 328.656 | 0.02304 | 0.203 | 0.783 | 135.00 |
| [Z 110-4](../data/galaxy_sources/z_110_4.json) | spiral | 101.9205 | 328.656 | 0.02304 | 0.766 | 0.294 | 83.00 |
| [MCG+00-08-082](../data/galaxy_sources/mcg_00_08_082.json) | spiral | 101.9645 | 328.796 | 0.02305 | 1.081 | 0.317 | 40.00 |
| [LEDA 1110080](../data/galaxy_sources/leda_1110080.json) | spiral | 101.9645 | 328.796 | 0.02305 | 0.572 | 0.247 | 84.00 |
| [Z 432-19](../data/galaxy_sources/z_432_19.json) | spiral | 101.9706 | 328.815 | 0.02305138 | 0.863 | 0.557 | 132.00 |
| [Z 69-39](../data/galaxy_sources/z_69_39.json) | spiral | 102.0085 | 328.936 | 0.02306 | 1.210 | 0.403 | 44.00 |
| [Z 45-72](../data/galaxy_sources/z_45_72.json) | spiral | 102.0085 | 328.936 | 0.02306 | 0.720 | 0.454 | 46.00 |
| [LEDA 39095](../data/galaxy_sources/leda_39095.json) | spiral | 102.0525 | 329.076 | 0.02307 | 0.539 | 0.549 | 90.00 |
| [Z 382-12](../data/galaxy_sources/z_382_12.json) | lenticular | 102.0607 | 329.102 | 0.02307186 | 0.716 | 0.672 | 107.00 |
| [NGC 4493](../data/galaxy_sources/ngc_4493.json) | lenticular | 102.0965 | 329.216 | 0.02308 | 1.756 | 0.566 | 135.00 |
| [Z 94-71](../data/galaxy_sources/z_94_71.json) | lenticular | 102.1405 | 329.357 | 0.02309 | 0.260 | 0.838 | 140.00 |
| [LEDA 1170468](../data/galaxy_sources/leda_1170468.json) | elliptical | 102.1405 | 329.357 | 0.02309 | 0.566 | 0.792 | 139.00 |
| [Z 402-14](../data/galaxy_sources/z_402_14.json) | spiral | 102.1845 | 329.497 | 0.0231 | 0.581 | 0.769 | 39.00 |
| [Z 64-77](../data/galaxy_sources/z_64_77.json) | spiral | 102.2285 | 329.637 | 0.02311 | 0.783 | 0.226 | 25.00 |
| [NGC 3535](../data/galaxy_sources/ngc_3535.json) | spiral | 102.2285 | 329.637 | 0.02311 | 1.656 | 0.456 | 177.00 |
| [Z 45-42](../data/galaxy_sources/z_45_42.json) | spiral | 102.2725 | 329.777 | 0.02312 | 0.812 | 0.226 | 130.00 |
| [UGC 11941](../data/galaxy_sources/ugc_11941.json) | spiral | 102.3160 | 329.916 | 0.02312991 | 0.843 | 0.241 | 169.00 |
| [MCG+00-60-032](../data/galaxy_sources/mcg_00_60_032.json) | spiral | 102.3604 | 330.058 | 0.02314 | 0.596 | 0.949 | 165.00 |
| [Z 41-15](../data/galaxy_sources/z_41_15.json) | spiral | 102.3604 | 330.058 | 0.02314 | 0.475 | 0.894 | 56.00 |
| [ESO 172-12](../data/galaxy_sources/eso_172_12.json) | barred spiral | 102.4317 | 330.285 | 0.0231562 | 0.720 | 0.781 | 15.00 |
| [UGC 12458](../data/galaxy_sources/ugc_12458.json) | spiral | 102.4362 | 330.299 | 0.02315723 | 1.103 | 0.147 | 136.00 |
| [Z 389-75](../data/galaxy_sources/z_389_75.json) | weakly barred spiral | 102.4484 | 330.338 | 0.02316 | 1.522 | 0.189 | 174.00 |
| [Z 122-84](../data/galaxy_sources/z_122_84.json) | spiral | 102.4484 | 330.338 | 0.02316 | 1.203 | 0.189 | 107.00 |
| [IC 3449](../data/galaxy_sources/ic_3449.json) | spiral | 102.4484 | 330.338 | 0.02316 | 0.325 | 0.586 | 143.00 |
| [Z 69-14](../data/galaxy_sources/z_69_14.json) | spiral | 102.4924 | 330.479 | 0.02317 | 0.564 | 0.765 | 164.00 |
| [IC 2999](../data/galaxy_sources/ic_2999.json) | spiral | 102.5364 | 330.619 | 0.02318 | 0.766 | 0.334 | 52.00 |
| [Z 374-27](../data/galaxy_sources/z_374_27.json) | spiral | 102.5364 | 330.619 | 0.02318 | 0.709 | 0.564 | 42.00 |
| [Z 80-51](../data/galaxy_sources/z_80_51.json) | spiral | 102.5364 | 330.619 | 0.02318 | 0.968 | 0.332 | 155.00 |
| [IC 4229](../data/galaxy_sources/ic_4229.json) | barred spiral | 102.5804 | 330.759 | 0.02319 | 0.811 | 0.944 | 41.00 |
| [MCG+06-27-031](../data/galaxy_sources/mcg_06_27_031.json) | barred spiral | 102.6244 | 330.899 | 0.0232 | 0.628 | 0.952 | 169.00 |
| [Z 45-118](../data/galaxy_sources/z_45_118.json) | spiral | 102.6244 | 330.899 | 0.0232 | 0.505 | 0.934 | 17.00 |
| [Z 268-37](../data/galaxy_sources/z_268_37.json) | elliptical | 102.6684 | 331.039 | 0.02321 | 0.660 | 0.926 | 4.00 |
| [Z 45-64](../data/galaxy_sources/z_45_64.json) | spiral | 102.6684 | 331.039 | 0.02321 | 0.500 | 0.904 | 125.00 |
| [Z 291-13](../data/galaxy_sources/z_291_13.json) | barred spiral | 102.7123 | 331.180 | 0.02322 | 0.609 | 0.661 | 138.00 |
| [LEDA 38631](../data/galaxy_sources/leda_38631.json) | spiral | 102.7563 | 331.320 | 0.02323 | 0.347 | 0.839 | 105.00 |
| [LEDA 31516](../data/galaxy_sources/leda_31516.json) | spiral | 102.8443 | 331.600 | 0.02325 | 0.516 | 0.745 | 20.00 |
| [Z 497-53](../data/galaxy_sources/z_497_53.json) | spiral | 102.9319 | 331.880 | 0.02326992 | 0.920 | 0.311 | 120.00 |
| [UGC 9988](../data/galaxy_sources/ugc_9988.json) | spiral | 102.9323 | 331.881 | 0.02327 | 1.122 | 0.186 | 67.00 |
| [LEDA 196744](../data/galaxy_sources/leda_196744.json) | spiral | 102.9323 | 331.881 | 0.02327 | 0.614 | 0.849 | 21.00 |
| [Z 460-42](../data/galaxy_sources/z_460_42.json) | spiral | 102.9620 | 331.975 | 0.02327675 | 0.940 | 0.429 | 14.00 |
| [Z 498-63](../data/galaxy_sources/z_498_63.json) | barred spiral | 102.9920 | 332.071 | 0.02328358 | 1.280 | 0.477 | 43.00 |
| [Z 290-44](../data/galaxy_sources/z_290_44.json) | barred spiral | 103.0642 | 332.301 | 0.0233 | 1.361 | 0.312 | 164.00 |
| [Z 268-24](../data/galaxy_sources/z_268_24.json) | spiral | 103.0642 | 332.301 | 0.0233 | 1.410 | 0.480 | 58.00 |
| [NGC 4288A](../data/galaxy_sources/ngc_4288a.json) | spiral | 103.0642 | 332.301 | 0.0233 | 0.567 | 0.962 | 89.00 |
| [Z 180-56](../data/galaxy_sources/z_180_56.json) | lenticular | 103.0642 | 332.301 | 0.0233 | 0.343 | 0.720 | 95.00 |
| [Z 42-78](../data/galaxy_sources/z_42_78.json) | spiral | 103.0642 | 332.301 | 0.0233 | 0.745 | 0.245 | 22.00 |
| [Z 43-101](../data/galaxy_sources/z_43_101.json) | spiral | 103.1522 | 332.582 | 0.02332 | 1.032 | 0.201 | 61.00 |
| [Z 381-21](../data/galaxy_sources/z_381_21.json) | elliptical | 103.1962 | 332.722 | 0.02333 | 0.665 | 0.705 | 59.00 |
| [LEDA 89685](../data/galaxy_sources/leda_89685.json) | spiral | 103.2842 | 333.002 | 0.02335 | 0.702 | 0.206 | 65.00 |
| [Z 170-11](../data/galaxy_sources/z_170_11.json) | elliptical | 103.2842 | 333.002 | 0.02335 | 0.647 | 0.834 | 169.00 |
| [ESO 39-1](../data/galaxy_sources/eso_39_1.json) | spiral | 103.3015 | 333.057 | 0.02335393 | 0.373 | 0.381 | 175.00 |
| [Z 299-46](../data/galaxy_sources/z_299_46.json) | spiral | 103.3282 | 333.143 | 0.02336 | 0.522 | 0.774 | 64.00 |
| [Z 211-43](../data/galaxy_sources/z_211_43.json) | spiral | 103.3282 | 333.143 | 0.02336 | 1.586 | 0.240 | 136.00 |
| [IC 643](../data/galaxy_sources/ic_643.json) | spiral | 103.3282 | 333.143 | 0.02336 | 1.073 | 0.366 | 73.00 |
| [IC 4030](../data/galaxy_sources/ic_4030.json) | barred lenticular | 103.3282 | 333.143 | 0.02336 | 0.447 | 0.650 | 12.00 |
| [Z 39-47](../data/galaxy_sources/z_39_47.json) | lenticular | 103.3721 | 333.283 | 0.02337 | 0.421 | 0.699 | 87.00 |
| [LEDA 49664](../data/galaxy_sources/leda_49664.json) | spiral | 103.3721 | 333.283 | 0.02337 | 0.424 | 0.522 | 168.00 |
| [Z 45-105](../data/galaxy_sources/z_45_105.json) | spiral | 103.4161 | 333.423 | 0.02338 | 0.634 | 0.441 | 51.00 |
| [Z 415-55](../data/galaxy_sources/z_415_55.json) | spiral | 103.4577 | 333.555 | 0.02338945 | 0.576 | 0.790 | 83.00 |
| [LEDA 38732](../data/galaxy_sources/leda_38732.json) | spiral | 103.4601 | 333.563 | 0.02339 | 0.508 | 0.400 | 32.00 |
| [Z 45-46](../data/galaxy_sources/z_45_46.json) | spiral | 103.5041 | 333.703 | 0.0234 | 0.960 | 0.306 | 5.00 |
| [LEDA 83720](../data/galaxy_sources/leda_83720.json) | barred lenticular | 103.5921 | 333.984 | 0.02342 | 0.503 | 0.649 | 77.00 |
| [Z 180-61](../data/galaxy_sources/z_180_61.json) | spiral | 103.6361 | 334.124 | 0.02343 | 0.784 | 0.374 | 1.00 |
| [Z 15-11](../data/galaxy_sources/z_15_11.json) | spiral | 103.6361 | 334.124 | 0.02343 | 0.544 | 0.483 | 94.00 |
| [Z 432-17](../data/galaxy_sources/z_432_17.json) | spiral | 103.6800 | 334.264 | 0.02344 | 0.727 | 0.458 | 138.00 |
| [NGC 4109](../data/galaxy_sources/ngc_4109.json) | spiral | 103.6800 | 334.264 | 0.02344 | 0.738 | 0.766 | 144.00 |
| [LEDA 47556](../data/galaxy_sources/leda_47556.json) | barred spiral | 103.6800 | 334.264 | 0.02344 | 0.493 | 0.665 | 21.00 |
| [Z 494-16](../data/galaxy_sources/z_494_16.json) | spiral | 103.6981 | 334.321 | 0.0234441 | 0.734 | 0.429 | 9.00 |
| [Z 499-82](../data/galaxy_sources/z_499_82.json) | spiral | 103.6981 | 334.321 | 0.0234441 | 0.783 | 0.887 | 174.00 |
| [MCG-02-60-009](../data/galaxy_sources/mcg_02_60_009.json) | barred spiral | 103.7240 | 334.404 | 0.02345 | 0.759 | 0.885 | 54.00 |
| [MCG+06-27-021](../data/galaxy_sources/mcg_06_27_021.json) | barred spiral | 103.7240 | 334.404 | 0.02345 | 0.513 | 0.852 | 62.00 |
| [Z 64-106](../data/galaxy_sources/z_64_106.json) | spiral | 103.7680 | 334.544 | 0.02346 | 0.755 | 0.730 | 96.00 |
| [Z 73-77](../data/galaxy_sources/z_73_77.json) | spiral | 103.7680 | 334.544 | 0.02346 | 0.744 | 0.513 | 37.00 |
| [Z 498-41](../data/galaxy_sources/z_498_41.json) | spiral | 103.7912 | 334.618 | 0.02346528 | 0.723 | 0.733 | 57.00 |
| [LEDA 23005](../data/galaxy_sources/leda_23005.json) | spiral | 103.8120 | 334.684 | 0.02347 | 0.435 | 0.753 | 61.00 |
| [Z 148-6](../data/galaxy_sources/z_148_6.json) | spiral | 103.8120 | 334.684 | 0.02347 | 0.594 | 0.944 | 9.00 |
| [LEDA 69532](../data/galaxy_sources/leda_69532.json) | spiral | 103.8183 | 334.704 | 0.02347143 | 0.426 | 0.834 | 146.00 |
| [MCG+04-08-002](../data/galaxy_sources/mcg_04_08_002.json) | spiral | 103.8183 | 334.704 | 0.02347143 | 0.343 | 0.580 | 70.00 |
| [LEDA 38758](../data/galaxy_sources/leda_38758.json) | spiral | 103.8560 | 334.825 | 0.02348 | 0.651 | 0.276 | 41.00 |
| [Z 42-50](../data/galaxy_sources/z_42_50.json) | spiral | 103.8560 | 334.825 | 0.02348 | 0.604 | 0.325 | 120.00 |
| [Z 45-77](../data/galaxy_sources/z_45_77.json) | spiral | 103.9000 | 334.965 | 0.02349 | 0.971 | 0.158 | 2.00 |
| [Z 74-97](../data/galaxy_sources/z_74_97.json) | spiral | 103.9000 | 334.965 | 0.02349 | 0.657 | 0.454 | 1.00 |
| [Z 93-36](../data/galaxy_sources/z_93_36.json) | spiral | 103.9385 | 335.087 | 0.02349875 | 2.148 | 0.116 | 176.00 |
| [IC 731](../data/galaxy_sources/ic_731.json) | spiral | 103.9439 | 335.105 | 0.0235 | 0.576 | 0.581 | 80.00 |
| [Z 241-48](../data/galaxy_sources/z_241_48.json) | spiral | 103.9879 | 335.245 | 0.02351 | 0.729 | 0.312 | 93.00 |
| [Z 147-27](../data/galaxy_sources/z_147_27.json) | spiral | 103.9985 | 335.279 | 0.02351242 | 0.927 | 0.337 | 178.00 |
| [UGC 12441](../data/galaxy_sources/ugc_12441.json) | barred spiral | 104.0286 | 335.375 | 0.02351925 | 0.878 | 0.244 | 15.00 |
| [MCG+06-20-046](../data/galaxy_sources/mcg_06_20_046.json) | elliptical | 104.0319 | 335.385 | 0.02352 | 0.646 | 0.872 | 31.00 |
| [Z 268-90](../data/galaxy_sources/z_268_90.json) | spiral | 104.0759 | 335.525 | 0.02353 | 0.666 | 0.471 | 50.00 |
| [Z 65-70](../data/galaxy_sources/z_65_70.json) | lenticular | 104.0759 | 335.525 | 0.02353 | 0.673 | 0.404 | 78.00 |
| [Z 500-69](../data/galaxy_sources/z_500_69.json) | spiral | 104.1037 | 335.614 | 0.02353633 | 0.643 | 0.603 | 40.00 |
| [Z 432-18](../data/galaxy_sources/z_432_18.json) | spiral | 104.1187 | 335.662 | 0.02353974 | 1.488 | 0.405 | 66.00 |
| [Z 266-59](../data/galaxy_sources/z_266_59.json) | spiral | 104.1199 | 335.666 | 0.02354 | 1.635 | 0.325 | 155.00 |
| [Z 162-15](../data/galaxy_sources/z_162_15.json) | spiral | 104.1199 | 335.666 | 0.02354 | 0.976 | 0.464 | 139.00 |
| [Z 169-48](../data/galaxy_sources/z_169_48.json) | spiral | 104.1199 | 335.666 | 0.02354 | 0.340 | 0.894 | 150.00 |
| [MCG+08-23-032](../data/galaxy_sources/mcg_08_23_032.json) | spiral | 104.1638 | 335.806 | 0.02355 | 1.020 | 0.502 | 90.00 |
| [Z 498-69](../data/galaxy_sources/z_498_69.json) | barred spiral | 104.2089 | 335.949 | 0.02356024 | 0.829 | 0.658 | 97.00 |
| [Z 476-61](../data/galaxy_sources/z_476_61.json) | spiral | 104.2239 | 335.997 | 0.02356366 | 0.566 | 0.758 | 92.00 |
| [Z 238-14](../data/galaxy_sources/z_238_14.json) | lenticular | 104.2518 | 336.086 | 0.02357 | 0.770 | 0.412 | 91.00 |
| [MCG+08-22-052](../data/galaxy_sources/mcg_08_22_052.json) | spiral | 104.2518 | 336.086 | 0.02357 | 0.487 | 0.730 | 109.00 |
| [Z 500-48](../data/galaxy_sources/z_500_48.json) | spiral | 104.2690 | 336.141 | 0.02357391 | 0.649 | 0.640 | 118.00 |
| [Z 43-32](../data/galaxy_sources/z_43_32.json) | spiral | 104.2958 | 336.226 | 0.02358 | 0.729 | 0.225 | 44.00 |
| [LEDA 38917](../data/galaxy_sources/leda_38917.json) | spiral | 104.3398 | 336.366 | 0.02359 | 0.451 | 0.469 | 173.00 |
| [Z 14-94](../data/galaxy_sources/z_14_94.json) | spiral | 104.3398 | 336.366 | 0.02359 | 0.827 | 0.426 | 51.00 |
| [Z 484-24](../data/galaxy_sources/z_484_24.json) | spiral | 104.3591 | 336.428 | 0.0235944 | 0.347 | 0.778 | 140.00 |
| [LEDA 80356](../data/galaxy_sources/leda_80356.json) | spiral | 104.3837 | 336.506 | 0.0236 | 0.595 | 0.360 | 7.00 |
| [Z 436-56](../data/galaxy_sources/z_436_56.json) | lenticular | 104.4277 | 336.647 | 0.02361 | 0.707 | 0.483 | 162.00 |
| [LEDA 86883](../data/galaxy_sources/leda_86883.json) | spiral | 104.4277 | 336.647 | 0.02361 | 0.457 | 0.876 | 173.00 |
| [Z 43-36](../data/galaxy_sources/z_43_36.json) | spiral | 104.4277 | 336.647 | 0.02361 | 0.443 | 0.851 | 103.00 |
| [Z 9-19](../data/galaxy_sources/z_9_19.json) | lenticular | 104.4493 | 336.715 | 0.0236149 | 0.612 | 0.503 | 175.00 |
| [Z 270-43](../data/galaxy_sources/z_270_43.json) | weakly barred spiral | 104.4717 | 336.787 | 0.02362 | 0.728 | 0.896 | 114.00 |
| [Z 180-45](../data/galaxy_sources/z_180_45.json) | spiral | 104.4717 | 336.787 | 0.02362 | 0.499 | 0.752 | 154.00 |
| [LEDA 82356](../data/galaxy_sources/leda_82356.json) | spiral | 104.4717 | 336.787 | 0.02362 | 0.415 | 0.677 | 2.00 |
| [Z 39-86](../data/galaxy_sources/z_39_86.json) | spiral | 104.4717 | 336.787 | 0.02362 | 0.474 | 0.485 | 4.00 |
| [MCG+07-21-031](../data/galaxy_sources/mcg_07_21_031.json) | spiral | 104.5157 | 336.927 | 0.02363 | 1.095 | 0.484 | 131.00 |
| [Z 148-11](../data/galaxy_sources/z_148_11.json) | spiral | 104.5597 | 337.067 | 0.02364 | 0.447 | 0.854 | 74.00 |
| [Z 498-20](../data/galaxy_sources/z_498_20.json) | spiral | 104.5845 | 337.146 | 0.02364565 | 0.679 | 0.877 | 15.00 |
| [NGC 4932](../data/galaxy_sources/ngc_4932.json) | weakly barred spiral | 104.6036 | 337.207 | 0.02365 | 1.010 | 0.927 | 164.00 |
| [Z 208-54](../data/galaxy_sources/z_208_54.json) | elliptical | 104.6036 | 337.207 | 0.02365 | 0.865 | 0.794 | 68.00 |
| [Z 101-23](../data/galaxy_sources/z_101_23.json) | spiral | 104.6036 | 337.207 | 0.02365 | 0.901 | 0.953 | 53.00 |
| [Z 374-24](../data/galaxy_sources/z_374_24.json) | spiral | 104.6036 | 337.207 | 0.02365 | 0.197 | 0.980 | 65.00 |
| [Z 389-73](../data/galaxy_sources/z_389_73.json) | barred spiral | 104.6145 | 337.242 | 0.02365248 | 1.044 | 0.304 | 79.00 |
| [IC 3039](../data/galaxy_sources/ic_3039.json) | spiral | 104.6296 | 337.290 | 0.0236559 | 0.972 | 0.420 | 20.00 |
| [Mrk 96](../data/galaxy_sources/mrk_96.json) | elliptical | 104.6476 | 337.347 | 0.02366 | 0.355 | 0.854 | 28.00 |
| [Z 102-13](../data/galaxy_sources/z_102_13.json) | spiral | 104.6476 | 337.347 | 0.02366 | 0.538 | 0.522 | 162.00 |
| [Z 501-19](../data/galaxy_sources/z_501_19.json) | spiral | 104.6596 | 337.385 | 0.02366273 | 0.502 | 0.735 | 137.00 |
| [Z 294-44](../data/galaxy_sources/z_294_44.json) | spiral | 104.6916 | 337.487 | 0.02367 | 0.635 | 0.759 | 69.00 |
| [Z 69-136](../data/galaxy_sources/z_69_136.json) | lenticular | 104.6916 | 337.487 | 0.02367 | 0.578 | 0.583 | 139.00 |
| [MCG+11-12-033](../data/galaxy_sources/mcg_11_12_033.json) | weakly barred spiral | 104.7356 | 337.627 | 0.02368 | 0.948 | 0.258 | 61.00 |
| [Z 429-11](../data/galaxy_sources/z_429_11.json) | spiral | 104.7796 | 337.768 | 0.02369 | 0.893 | 0.212 | 156.00 |
| [LEDA 34983](../data/galaxy_sources/leda_34983.json) | spiral | 104.8235 | 337.908 | 0.0237 | 0.592 | 0.381 | 29.00 |
| [Z 454-41](../data/galaxy_sources/z_454_41.json) | spiral | 104.8399 | 337.960 | 0.02370373 | 0.829 | 0.300 | 166.00 |
| [LEDA 37129](../data/galaxy_sources/leda_37129.json) | spiral | 104.8399 | 337.960 | 0.02370373 | 0.347 | 0.535 | 140.00 |
| [Z 244-17](../data/galaxy_sources/z_244_17.json) | spiral | 104.8675 | 338.048 | 0.02371 | 0.724 | 0.923 | 75.00 |
| [LEDA 34862](../data/galaxy_sources/leda_34862.json) | spiral | 104.8675 | 338.048 | 0.02371 | 0.556 | 0.538 | 102.00 |
| [Z 45-122](../data/galaxy_sources/z_45_122.json) | spiral | 104.8675 | 338.048 | 0.02371 | 0.558 | 0.665 | 123.00 |
| [Z 496-45](../data/galaxy_sources/z_496_45.json) | barred spiral | 104.8850 | 338.103 | 0.02371398 | 0.912 | 0.799 | 174.00 |
| [IC 3713](../data/galaxy_sources/ic_3713.json) | lenticular | 104.9115 | 338.188 | 0.02372 | 0.763 | 0.664 | 132.00 |
| [Z 45-87](../data/galaxy_sources/z_45_87.json) | weakly barred spiral | 104.9451 | 338.295 | 0.02372764 | 0.390 | 0.641 | 155.00 |
| [Z 77-123](../data/galaxy_sources/z_77_123.json) | spiral | 104.9451 | 338.295 | 0.02372764 | 0.464 | 0.979 | 109.00 |
| [Z 156-66](../data/galaxy_sources/z_156_66.json) | spiral | 104.9555 | 338.328 | 0.02373 | 0.835 | 0.735 | 159.00 |
| [Z 215-19](../data/galaxy_sources/z_215_19.json) | elliptical | 104.9555 | 338.328 | 0.02373 | 0.688 | 0.888 | 170.00 |
| [NGC 5083](../data/galaxy_sources/ngc_5083.json) | barred spiral | 104.9555 | 338.328 | 0.02373 | 1.112 | 0.783 | 137.00 |
| [Z 106-41](../data/galaxy_sources/z_106_41.json) | spiral | 104.9902 | 338.439 | 0.02373789 | 0.919 | 0.872 | 159.00 |
| [Z 35-42](../data/galaxy_sources/z_35_42.json) | spiral | 105.0434 | 338.608 | 0.02375 | 0.612 | 0.380 | 142.00 |
| [LEDA 86367](../data/galaxy_sources/leda_86367.json) | spiral | 105.0434 | 338.608 | 0.02375 | 0.605 | 0.262 | 106.00 |
| [MCG+08-16-002](../data/galaxy_sources/mcg_08_16_002.json) | spiral | 105.0874 | 338.748 | 0.02376 | 0.743 | 0.613 | 73.00 |
| [Z 64-51](../data/galaxy_sources/z_64_51.json) | lenticular | 105.0874 | 338.748 | 0.02376 | 0.647 | 0.548 | 20.00 |
| [Z 212-61](../data/galaxy_sources/z_212_61.json) | spiral | 105.0874 | 338.748 | 0.02376 | 0.785 | 0.892 | 93.00 |
| [Z 39-110](../data/galaxy_sources/z_39_110.json) | spiral | 105.0874 | 338.748 | 0.02376 | 0.630 | 0.272 | 156.00 |
| [Z 214-24](../data/galaxy_sources/z_214_24.json) | spiral | 105.0874 | 338.748 | 0.02376 | 1.118 | 0.372 | 56.00 |
| [LEDA 37113](../data/galaxy_sources/leda_37113.json) | spiral | 105.0874 | 338.748 | 0.02376 | 0.415 | 0.677 | 65.00 |
| [Z 45-123](../data/galaxy_sources/z_45_123.json) | spiral | 105.0874 | 338.748 | 0.02376 | 0.831 | 0.525 | 173.00 |
| [LEDA 59008](../data/galaxy_sources/leda_59008.json) | spiral | 105.0874 | 338.748 | 0.02376 | 0.268 | 0.838 | 30.00 |
| [IC 4301](../data/galaxy_sources/ic_4301.json) | spiral | 105.1705 | 339.013 | 0.02377889 | 1.205 | 0.306 | 130.00 |
| [LEDA 1193311](../data/galaxy_sources/leda_1193311.json) | spiral | 105.1754 | 339.029 | 0.02378 | 0.597 | 0.294 | 88.00 |
| [Z 493-12](../data/galaxy_sources/z_493_12.json) | spiral | 105.1930 | 339.085 | 0.02378402 | 0.443 | 0.700 | 160.00 |
| [Z 497-20](../data/galaxy_sources/z_497_20.json) | spiral | 105.2005 | 339.109 | 0.02378573 | 0.553 | 0.956 | 74.00 |
| [Z 237-4](../data/galaxy_sources/z_237_4.json) | spiral | 105.2193 | 339.169 | 0.02379 | 0.751 | 0.652 | 155.00 |
| [MCG+07-25-018](../data/galaxy_sources/mcg_07_25_018.json) | spiral | 105.2193 | 339.169 | 0.02379 | 0.643 | 0.829 | 138.00 |
| [LEDA 91190](../data/galaxy_sources/leda_91190.json) | spiral | 105.2193 | 339.169 | 0.02379 | 0.971 | 0.178 | 59.00 |
| [Z 476-65](../data/galaxy_sources/z_476_65.json) | spiral | 105.2606 | 339.300 | 0.0237994 | 0.472 | 0.814 | 29.00 |
| [Z 64-42](../data/galaxy_sources/z_64_42.json) | spiral | 105.2633 | 339.309 | 0.0238 | 0.574 | 0.531 | 60.00 |
| [LEDA 56496](../data/galaxy_sources/leda_56496.json) | spiral | 105.3073 | 339.449 | 0.02381 | 0.240 | 0.921 | 155.00 |
| [MCG+04-26-012](../data/galaxy_sources/mcg_04_26_012.json) | elliptical | 105.3073 | 339.449 | 0.02381 | 0.967 | 0.758 | 52.00 |
| [NGC 5270](../data/galaxy_sources/ngc_5270.json) | weakly barred spiral | 105.3809 | 339.683 | 0.02382673 | 1.049 | 0.705 | 9.00 |
| [Z 266-46](../data/galaxy_sources/z_266_46.json) | spiral | 105.3952 | 339.729 | 0.02383 | 0.721 | 0.979 | 43.00 |
| [NGC 3644](../data/galaxy_sources/ngc_3644.json) | spiral | 105.3952 | 339.729 | 0.02383 | 1.275 | 0.465 | 62.00 |
| [NGC 4963](../data/galaxy_sources/ngc_4963.json) | spiral | 105.3952 | 339.729 | 0.02383 | 0.895 | 0.955 | 14.00 |
| [Z 102-30](../data/galaxy_sources/z_102_30.json) | spiral | 105.4710 | 339.970 | 0.02384723 | 1.479 | 0.197 | 138.00 |
| [MCG+08-23-021](../data/galaxy_sources/mcg_08_23_021.json) | spiral | 105.4832 | 340.009 | 0.02385 | 0.590 | 0.756 | 119.00 |
| [NGC 3179](../data/galaxy_sources/ngc_3179.json) | spiral | 105.4832 | 340.009 | 0.02385 | 1.609 | 0.411 | 47.00 |
| [LEDA 1020235](../data/galaxy_sources/leda_1020235.json) | spiral | 105.4832 | 340.009 | 0.02385 | 0.996 | 0.184 | 163.00 |
| [LEDA 10082](../data/galaxy_sources/leda_10082.json) | spiral | 105.5272 | 340.149 | 0.02386 | 0.495 | 0.778 | 77.00 |
| [LEDA 197390](../data/galaxy_sources/leda_197390.json) | lenticular | 105.5272 | 340.149 | 0.02386 | 0.517 | 0.618 | 180.00 |
| [Z 71-61](../data/galaxy_sources/z_71_61.json) | spiral | 105.5272 | 340.149 | 0.02386 | 0.622 | 0.592 | 61.00 |
| [LEDA 29941](../data/galaxy_sources/leda_29941.json) | spiral | 105.5711 | 340.289 | 0.02387 | 0.496 | 0.964 | 104.00 |
| [Z 148-24](../data/galaxy_sources/z_148_24.json) | spiral | 105.5912 | 340.353 | 0.02387457 | 0.394 | 0.745 | 124.00 |
| [Z 198-12](../data/galaxy_sources/z_198_12.json) | spiral | 105.6591 | 340.569 | 0.02389 | 0.545 | 0.814 | 152.00 |
| [Z 475-39](../data/galaxy_sources/z_475_39.json) | spiral | 105.6814 | 340.641 | 0.02389507 | 0.899 | 0.340 | 59.00 |
| [Z 263-81](../data/galaxy_sources/z_263_81.json) | barred spiral | 105.7470 | 340.850 | 0.02391 | 0.728 | 0.597 | 24.00 |
| [Z 494-9](../data/galaxy_sources/z_494_9.json) | spiral | 105.8317 | 341.119 | 0.02392925 | 0.935 | 0.494 | 23.00 |
| [Z 271-11](../data/galaxy_sources/z_271_11.json) | spiral | 105.8350 | 341.130 | 0.02393 | 0.836 | 0.780 | 67.00 |
| [Z 267-17](../data/galaxy_sources/z_267_17.json) | barred spiral | 105.8350 | 341.130 | 0.02393 | 0.513 | 0.741 | 36.00 |
| [UGC 12435](../data/galaxy_sources/ugc_12435.json) | barred spiral | 105.9218 | 341.406 | 0.02394975 | 0.699 | 0.638 | 45.00 |
| [MCG+09-19-074](../data/galaxy_sources/mcg_09_19_074.json) | spiral | 105.9229 | 341.410 | 0.02395 | 0.480 | 0.618 | 17.00 |
| [Z 122-87](../data/galaxy_sources/z_122_87.json) | spiral | 105.9519 | 341.502 | 0.02395659 | 0.856 | 0.732 | 134.00 |
| [Z 241-69](../data/galaxy_sources/z_241_69.json) | spiral | 105.9669 | 341.550 | 0.02396 | 0.618 | 0.943 | 151.00 |
| [Z 42-190](../data/galaxy_sources/z_42_190.json) | spiral | 105.9669 | 341.550 | 0.02396 | 0.655 | 0.441 | 85.00 |
| [Z 71-84](../data/galaxy_sources/z_71_84.json) | spiral | 105.9669 | 341.550 | 0.02396 | 0.869 | 0.288 | 50.00 |
| [Z 35-34](../data/galaxy_sources/z_35_34.json) | spiral | 106.0109 | 341.690 | 0.02397 | 0.499 | 0.754 | 39.00 |
| [Z 215-60](../data/galaxy_sources/z_215_60.json) | lenticular | 106.0988 | 341.970 | 0.02399 | 0.758 | 0.427 | 25.00 |
| [Z 66-31](../data/galaxy_sources/z_66_31.json) | spiral | 106.1428 | 342.110 | 0.024 | 0.602 | 0.413 | 68.00 |
| [Z 42-25](../data/galaxy_sources/z_42_25.json) | spiral | 106.1428 | 342.110 | 0.024 | 0.644 | 0.898 | 133.00 |
| [Z 46-35](../data/galaxy_sources/z_46_35.json) | spiral | 106.1923 | 342.268 | 0.02401127 | 1.330 | 0.348 | 170.00 |
| [UGC 12252](../data/galaxy_sources/ugc_12252.json) | spiral | 106.2224 | 342.364 | 0.0240181 | 1.385 | 0.165 | 172.00 |
| [LEDA 70161](../data/galaxy_sources/leda_70161.json) | spiral | 106.2307 | 342.390 | 0.02402 | 0.366 | 0.829 | 86.00 |
| [Z 244-38](../data/galaxy_sources/z_244_38.json) | spiral | 106.3187 | 342.670 | 0.02404 | 0.884 | 0.392 | 69.00 |
| [Z 36-37](../data/galaxy_sources/z_36_37.json) | lenticular | 106.3187 | 342.670 | 0.02404 | 0.631 | 0.459 | 31.00 |
| [LEDA 93697](../data/galaxy_sources/leda_93697.json) | elliptical | 106.3626 | 342.810 | 0.02405 | 0.325 | 0.550 | 104.00 |
| [Z 46-12](../data/galaxy_sources/z_46_12.json) | lenticular | 106.3626 | 342.810 | 0.02405 | 0.488 | 0.916 | 1.00 |
| [LEDA 25364](../data/galaxy_sources/leda_25364.json) | spiral | 106.4066 | 342.951 | 0.02406 | 0.415 | 0.900 | 17.00 |
| [LEDA 1985205](../data/galaxy_sources/leda_1985205.json) | elliptical | 106.4066 | 342.951 | 0.02406 | 0.273 | 0.925 | 103.00 |
| [LEDA 56638](../data/galaxy_sources/leda_56638.json) | spiral | 106.4066 | 342.951 | 0.02406 | 0.794 | 0.275 | 144.00 |
| [LEDA 12239](../data/galaxy_sources/leda_12239.json) | spiral | 106.4506 | 343.091 | 0.02407 | 0.790 | 0.238 | 29.00 |
| [Z 243-27](../data/galaxy_sources/z_243_27.json) | lenticular | 106.4506 | 343.091 | 0.02407 | 1.306 | 0.273 | 170.00 |
| [MCG+07-26-023](../data/galaxy_sources/mcg_07_26_023.json) | spiral | 106.4506 | 343.091 | 0.02407 | 0.829 | 0.605 | 10.00 |
| [LEDA 222](../data/galaxy_sources/leda_222.json) | spiral | 106.4628 | 343.130 | 0.02407279 | 0.613 | 0.495 | 10.00 |
| [LEDA 86892](../data/galaxy_sources/leda_86892.json) | spiral | 106.4945 | 343.231 | 0.02408 | 0.938 | 0.242 | 73.00 |
| [Z 71-96](../data/galaxy_sources/z_71_96.json) | spiral | 106.4945 | 343.231 | 0.02408 | 0.675 | 0.808 | 120.00 |
| [LEDA 1180356](../data/galaxy_sources/leda_1180356.json) | spiral | 106.4945 | 343.231 | 0.02408 | 0.328 | 0.934 | 71.00 |
| [Z 493-16](../data/galaxy_sources/z_493_16.json) | spiral | 106.5079 | 343.273 | 0.02408304 | 0.360 | 0.719 | 55.00 |
| [LEDA 1913295](../data/galaxy_sources/leda_1913295.json) | barred spiral | 106.5385 | 343.371 | 0.02409 | 0.424 | 0.693 | 97.00 |
| [LEDA 2534297](../data/galaxy_sources/leda_2534297.json) | lenticular | 106.6265 | 343.651 | 0.02411 | 0.355 | 0.820 | 76.00 |
| [Z 209-40](../data/galaxy_sources/z_209_40.json) | elliptical | 106.6265 | 343.651 | 0.02411 | 0.820 | 0.822 | 77.00 |
| [MCG+08-23-027](../data/galaxy_sources/mcg_08_23_027.json) | spiral | 106.6265 | 343.651 | 0.02411 | 1.543 | 0.166 | 58.00 |
| [Z 16-42](../data/galaxy_sources/z_16_42.json) | spiral | 106.6265 | 343.651 | 0.02411 | 0.587 | 0.881 | 43.00 |
| [Z 218-18](../data/galaxy_sources/z_218_18.json) | spiral | 106.6265 | 343.651 | 0.02411 | 0.745 | 0.598 | 49.00 |
| [Z 72-64](../data/galaxy_sources/z_72_64.json) | spiral | 106.6704 | 343.791 | 0.02412 | 0.828 | 0.434 | 151.00 |
| [IC 5254](../data/galaxy_sources/ic_5254.json) | spiral | 106.6732 | 343.800 | 0.02412064 | 0.870 | 0.385 | 80.00 |
| [Z 149-20](../data/galaxy_sources/z_149_20.json) | spiral | 106.6883 | 343.848 | 0.02412406 | 0.718 | 0.754 | 42.00 |
| [Z 74-123](../data/galaxy_sources/z_74_123.json) | barred spiral | 106.7144 | 343.931 | 0.02413 | 0.745 | 0.549 | 59.00 |
| [Z 473-5](../data/galaxy_sources/z_473_5.json) | spiral | 106.7484 | 344.039 | 0.02413773 | 0.485 | 0.865 | 14.00 |
| [Z 12-63](../data/galaxy_sources/z_12_63.json) | spiral | 106.7584 | 344.071 | 0.02414 | 0.495 | 0.631 | 93.00 |
| [IC 4012](../data/galaxy_sources/ic_4012.json) | elliptical | 106.7584 | 344.071 | 0.02414 | 0.403 | 0.796 | 132.00 |
| [MCG+07-28-007](../data/galaxy_sources/mcg_07_28_007.json) | weakly barred spiral | 106.7584 | 344.071 | 0.02414 | 0.749 | 0.711 | 54.00 |
| [Z 484-22](../data/galaxy_sources/z_484_22.json) | spiral | 106.7634 | 344.087 | 0.02414115 | 0.433 | 0.580 | 40.00 |
| [Z 101-49](../data/galaxy_sources/z_101_49.json) | spiral | 106.7784 | 344.135 | 0.02414457 | 0.964 | 0.788 | 54.00 |
| [IC 3271](../data/galaxy_sources/ic_3271.json) | weakly barred spiral | 106.8023 | 344.211 | 0.02415 | 0.826 | 0.939 | 3.00 |
| [Z 107-31](../data/galaxy_sources/z_107_31.json) | spiral | 106.8023 | 344.211 | 0.02415 | 0.699 | 0.902 | 153.00 |
| [Z 47-132](../data/galaxy_sources/z_47_132.json) | spiral | 106.8386 | 344.326 | 0.02415824 | 0.957 | 0.803 | 62.00 |
| [Z 42-166](../data/galaxy_sources/z_42_166.json) | lenticular | 106.8463 | 344.351 | 0.02416 | 0.890 | 0.303 | 121.00 |
| [Z 73-63](../data/galaxy_sources/z_73_63.json) | weakly barred spiral | 106.8836 | 344.470 | 0.02416849 | 0.932 | 0.189 | 8.00 |
| [MCG+10-16-076](../data/galaxy_sources/mcg_10_16_076.json) | spiral | 106.8903 | 344.491 | 0.02417 | 0.448 | 0.823 | 28.00 |
| [LEDA 12486](../data/galaxy_sources/leda_12486.json) | lenticular | 106.9342 | 344.631 | 0.02418 | 0.429 | 0.470 | 135.00 |
| [LEDA 32563](../data/galaxy_sources/leda_32563.json) | spiral | 106.9782 | 344.771 | 0.02419 | 0.804 | 0.277 | 131.00 |
| [Z 74-130](../data/galaxy_sources/z_74_130.json) | spiral | 106.9782 | 344.771 | 0.02419 | 0.663 | 0.276 | 4.00 |
| [Z 452-23](../data/galaxy_sources/z_452_23.json) | spiral | 107.0039 | 344.853 | 0.02419584 | 0.761 | 0.627 | 38.00 |
| [Z 236-10](../data/galaxy_sources/z_236_10.json) | weakly barred spiral | 107.0222 | 344.911 | 0.0242 | 0.923 | 0.761 | 158.00 |
| [Z 63-63](../data/galaxy_sources/z_63_63.json) | spiral | 107.0222 | 344.911 | 0.0242 | 0.629 | 0.863 | 169.00 |
| [NGC 831](../data/galaxy_sources/ngc_831.json) | spiral | 107.0490 | 344.996 | 0.02420609 | 0.629 | 0.786 | 109.00 |
| [LEDA 85283](../data/galaxy_sources/leda_85283.json) | spiral | 107.0640 | 345.044 | 0.02420951 | 0.303 | 0.640 | 150.00 |
| [LEDA 37283](../data/galaxy_sources/leda_37283.json) | spiral | 107.0661 | 345.051 | 0.02421 | 0.473 | 0.916 | 157.00 |
| [LEDA 39496](../data/galaxy_sources/leda_39496.json) | lenticular | 107.0661 | 345.051 | 0.02421 | 0.389 | 0.632 | 151.00 |
| [Z 239-47](../data/galaxy_sources/z_239_47.json) | spiral | 107.1101 | 345.191 | 0.02422 | 1.226 | 0.324 | 28.00 |
| [MCG+06-21-036](../data/galaxy_sources/mcg_06_21_036.json) | spiral | 107.1101 | 345.191 | 0.02422 | 0.260 | 0.681 | 165.00 |
| [LEDA 83524](../data/galaxy_sources/leda_83524.json) | barred spiral | 107.1101 | 345.191 | 0.02422 | 0.368 | 0.735 | 130.00 |
| [Z 70-166](../data/galaxy_sources/z_70_166.json) | spiral | 107.1101 | 345.191 | 0.02422 | 0.365 | 0.639 | 52.00 |
| [Z 462-34](../data/galaxy_sources/z_462_34.json) | barred spiral | 107.1391 | 345.283 | 0.0242266 | 0.721 | 0.677 | 151.00 |
| [Z 208-59](../data/galaxy_sources/z_208_59.json) | spiral | 107.1980 | 345.471 | 0.02424 | 0.473 | 0.697 | 84.00 |
| [MCG+08-23-072](../data/galaxy_sources/mcg_08_23_072.json) | spiral | 107.1980 | 345.471 | 0.02424 | 0.802 | 0.333 | 177.00 |
| [Z 117-68](../data/galaxy_sources/z_117_68.json) | spiral | 107.1980 | 345.471 | 0.02424 | 1.453 | 0.210 | 49.00 |
| [LEDA 45910](../data/galaxy_sources/leda_45910.json) | lenticular | 107.1980 | 345.471 | 0.02424 | 0.227 | 0.921 | 20.00 |
| [LEDA 2009009](../data/galaxy_sources/leda_2009009.json) | elliptical | 107.1980 | 345.471 | 0.02424 | 0.428 | 0.726 | 85.00 |
| [Z 244-35](../data/galaxy_sources/z_244_35.json) | spiral | 107.2420 | 345.611 | 0.02425 | 1.135 | 0.431 | 75.00 |
| [MCG+07-27-036](../data/galaxy_sources/mcg_07_27_036.json) | spiral | 107.2420 | 345.611 | 0.02425 | 0.830 | 0.302 | 175.00 |
| [IC 4011](../data/galaxy_sources/ic_4011.json) | elliptical | 107.2420 | 345.611 | 0.02425 | 0.287 | 0.899 | 15.00 |
| [IC 1882](../data/galaxy_sources/ic_1882.json) | spiral | 107.2443 | 345.618 | 0.02425053 | 1.380 | 0.383 | 24.00 |
| [Z 414-2](../data/galaxy_sources/z_414_2.json) | barred spiral | 107.3345 | 345.906 | 0.02427104 | 0.739 | 0.684 | 48.00 |
| [LEDA 2815934](../data/galaxy_sources/leda_2815934.json) | lenticular | 107.4179 | 346.171 | 0.02429 | 0.541 | 0.513 | 114.00 |
| [MCG+10-15-082](../data/galaxy_sources/mcg_10_15_082.json) | barred spiral | 107.4618 | 346.311 | 0.0243 | 0.499 | 0.808 | 159.00 |
| [Z 430-59](../data/galaxy_sources/z_430_59.json) | spiral | 107.4618 | 346.311 | 0.0243 | 1.030 | 0.471 | 22.00 |
| [UGC 6562](../data/galaxy_sources/ugc_6562.json) | barred spiral | 107.4618 | 346.311 | 0.0243 | 0.868 | 0.200 | 149.00 |
| [Z 42-195](../data/galaxy_sources/z_42_195.json) | spiral | 107.5058 | 346.451 | 0.02431 | 0.734 | 0.428 | 72.00 |
| [Z 241-90](../data/galaxy_sources/z_241_90.json) | spiral | 107.5937 | 346.731 | 0.02433 | 0.535 | 0.553 | 169.00 |
| [UGC 5235](../data/galaxy_sources/ugc_5235.json) | spiral | 107.5937 | 346.731 | 0.02433 | 1.527 | 0.133 | 151.00 |
| [Z 214-17](../data/galaxy_sources/z_214_17.json) | spiral | 107.5937 | 346.731 | 0.02433 | 0.717 | 0.533 | 37.00 |
| [Z 72-30](../data/galaxy_sources/z_72_30.json) | lenticular | 107.5937 | 346.731 | 0.02433 | 0.504 | 0.724 | 150.00 |
| [LEDA 2000577](../data/galaxy_sources/leda_2000577.json) | elliptical | 107.5937 | 346.731 | 0.02433 | 0.454 | 0.809 | 85.00 |
| [Z 496-17](../data/galaxy_sources/z_496_17.json) | spiral | 107.6351 | 346.863 | 0.02433942 | 1.349 | 0.363 | 131.00 |
| [IC 3107](../data/galaxy_sources/ic_3107.json) | spiral | 107.6377 | 346.871 | 0.02434 | 1.366 | 0.490 | 133.00 |
| [LEDA 93846](../data/galaxy_sources/leda_93846.json) | spiral | 107.6377 | 346.871 | 0.02434 | 0.877 | 0.288 | 124.00 |
| [Z 239-12](../data/galaxy_sources/z_239_12.json) | spiral | 107.6817 | 347.011 | 0.02435 | 0.672 | 0.696 | 58.00 |
| [Z 214-23](../data/galaxy_sources/z_214_23.json) | spiral | 107.7256 | 347.151 | 0.02436 | 0.649 | 0.671 | 115.00 |
| [Z 47-51](../data/galaxy_sources/z_47_51.json) | weakly barred spiral | 107.7256 | 347.151 | 0.02436 | 0.954 | 0.274 | 137.00 |
| [MCG+05-27-069](../data/galaxy_sources/mcg_05_27_069.json) | spiral | 107.8135 | 347.431 | 0.02438 | 0.565 | 0.842 | 161.00 |
| [Z 136-96](../data/galaxy_sources/z_136_96.json) | spiral | 107.8135 | 347.431 | 0.02438 | 0.773 | 0.502 | 123.00 |
| [Z 454-51](../data/galaxy_sources/z_454_51.json) | spiral | 107.8155 | 347.437 | 0.02438044 | 0.895 | 0.335 | 67.00 |
| [Z 387-43](../data/galaxy_sources/z_387_43.json) | elliptical | 107.8456 | 347.533 | 0.02438728 | 0.921 | 0.905 | 131.00 |
| [Z 208-53](../data/galaxy_sources/z_208_53.json) | spiral | 107.8575 | 347.571 | 0.02439 | 0.684 | 0.580 | 25.00 |
| [Z 208-66](../data/galaxy_sources/z_208_66.json) | spiral | 107.8575 | 347.571 | 0.02439 | 0.498 | 0.857 | 150.00 |
| [ESO 114-14](../data/galaxy_sources/eso_114_14.json) | spiral | 107.9102 | 347.739 | 0.02440198 | 1.206 | 0.904 | 83.00 |
| [Z 500-14](../data/galaxy_sources/z_500_14.json) | weakly barred spiral | 107.9207 | 347.772 | 0.02440438 | 0.701 | 0.590 | 26.00 |
| [ESO 60-10](../data/galaxy_sources/eso_60_10.json) | spiral | 107.9388 | 347.829 | 0.02440848 | 0.403 | 0.801 | 60.00 |
| [NGC 3238](../data/galaxy_sources/ngc_3238.json) | elliptical | 107.9454 | 347.851 | 0.02441 | 1.388 | 0.849 | 149.00 |
| [Z 151-46](../data/galaxy_sources/z_151_46.json) | lenticular | 107.9454 | 347.851 | 0.02441 | 0.729 | 0.927 | 34.00 |
| [LEDA 32152](../data/galaxy_sources/leda_32152.json) | spiral | 107.9454 | 347.851 | 0.02441 | 0.558 | 0.483 | 120.00 |
| [MCG+02-55-029](../data/galaxy_sources/mcg_02_55_029.json) | spiral | 107.9959 | 348.011 | 0.02442147 | 1.019 | 0.245 | 151.00 |
| [MCG+08-23-036](../data/galaxy_sources/mcg_08_23_036.json) | spiral | 108.0334 | 348.131 | 0.02443 | 0.498 | 0.738 | 112.00 |
| [Z 46-90](../data/galaxy_sources/z_46_90.json) | spiral | 108.0334 | 348.131 | 0.02443 | 0.621 | 0.576 | 169.00 |
| [Z 263-16](../data/galaxy_sources/z_263_16.json) | elliptical | 108.0773 | 348.271 | 0.02444 | 0.460 | 0.700 | 145.00 |
| [Z 248-53](../data/galaxy_sources/z_248_53.json) | spiral | 108.0773 | 348.271 | 0.02444 | 0.721 | 0.604 | 133.00 |
| [Z 208-11](../data/galaxy_sources/z_208_11.json) | spiral | 108.0773 | 348.271 | 0.02444 | 0.669 | 0.478 | 104.00 |
| [Z 244-28](../data/galaxy_sources/z_244_28.json) | spiral | 108.0773 | 348.271 | 0.02444 | 0.675 | 0.587 | 103.00 |
| [Z 11-80](../data/galaxy_sources/z_11_80.json) | spiral | 108.1213 | 348.411 | 0.02445 | 0.625 | 0.522 | 90.00 |
| [Z 452-21](../data/galaxy_sources/z_452_21.json) | spiral | 108.1462 | 348.490 | 0.02445566 | 0.666 | 0.508 | 61.00 |
| [Z 454-39](../data/galaxy_sources/z_454_39.json) | spiral | 108.1913 | 348.633 | 0.02446592 | 1.679 | 0.177 | 80.00 |
| [Z 101-38](../data/galaxy_sources/z_101_38.json) | spiral | 108.2063 | 348.681 | 0.02446934 | 1.101 | 0.131 | 177.00 |
| [Z 208-9](../data/galaxy_sources/z_208_9.json) | spiral | 108.2092 | 348.691 | 0.02447 | 0.607 | 0.640 | 156.00 |
| [Z 209-19](../data/galaxy_sources/z_209_19.json) | lenticular | 108.2092 | 348.691 | 0.02447 | 0.737 | 0.794 | 56.00 |
| [Z 68-86](../data/galaxy_sources/z_68_86.json) | spiral | 108.2092 | 348.691 | 0.02447 | 0.579 | 0.951 | 79.00 |
| [Z 495-26](../data/galaxy_sources/z_495_26.json) | barred spiral | 108.2213 | 348.729 | 0.02447276 | 2.366 | 0.140 | 171.00 |
| [Z 241-77](../data/galaxy_sources/z_241_77.json) | spiral | 108.2532 | 348.830 | 0.02448 | 0.946 | 0.450 | 11.00 |
| [Z 117-65](../data/galaxy_sources/z_117_65.json) | elliptical | 108.2971 | 348.970 | 0.02449 | 0.691 | 0.743 | 75.00 |
| [LEDA 2015943](../data/galaxy_sources/leda_2015943.json) | barred spiral | 108.2971 | 348.970 | 0.02449 | 0.618 | 0.444 | 59.00 |
| [Z 45-113](../data/galaxy_sources/z_45_113.json) | spiral | 108.2971 | 348.970 | 0.02449 | 0.538 | 0.582 | 164.00 |
| [MCG-02-58-014](../data/galaxy_sources/mcg_02_58_014.json) | spiral | 108.3411 | 349.110 | 0.0245 | 0.835 | 0.436 | 8.00 |
| [Z 17-89](../data/galaxy_sources/z_17_89.json) | lenticular | 108.3411 | 349.110 | 0.0245 | 0.498 | 0.696 | 141.00 |
| [Z 430-46](../data/galaxy_sources/z_430_46.json) | spiral | 108.3717 | 349.208 | 0.02450695 | 0.753 | 0.766 | 165.00 |
| [LEDA 9558](../data/galaxy_sources/leda_9558.json) | spiral | 108.3851 | 349.250 | 0.02451 | 0.381 | 0.509 | 108.00 |
| [MCG+00-08-005](../data/galaxy_sources/mcg_00_08_005.json) | spiral | 108.3851 | 349.250 | 0.02451 | 0.615 | 0.894 | 175.00 |
| [Z 122-94](../data/galaxy_sources/z_122_94.json) | spiral | 108.4290 | 349.390 | 0.02452 | 1.038 | 0.480 | 73.00 |
| [LEDA 86998](../data/galaxy_sources/leda_86998.json) | spiral | 108.4619 | 349.495 | 0.02452747 | 0.718 | 0.251 | 158.00 |
| [Z 70-15](../data/galaxy_sources/z_70_15.json) | spiral | 108.4730 | 349.530 | 0.02453 | 1.085 | 0.568 | 77.00 |
| [IC 1460](../data/galaxy_sources/ic_1460.json) | elliptical | 108.4769 | 349.543 | 0.02453089 | 0.737 | 0.900 | 164.00 |
| [MCG+09-17-045](../data/galaxy_sources/mcg_09_17_045.json) | barred spiral | 108.5169 | 349.670 | 0.02454 | 0.463 | 0.680 | 64.00 |
| [MCG+08-20-042](../data/galaxy_sources/mcg_08_20_042.json) | barred spiral | 108.5169 | 349.670 | 0.02454 | 1.091 | 0.199 | 79.00 |
| [Z 213-15](../data/galaxy_sources/z_213_15.json) | spiral | 108.5609 | 349.810 | 0.02455 | 0.649 | 0.902 | 94.00 |
| [Z 18-48](../data/galaxy_sources/z_18_48.json) | spiral | 108.5609 | 349.810 | 0.02455 | 0.718 | 0.856 | 119.00 |
| [LEDA 51215](../data/galaxy_sources/leda_51215.json) | spiral | 108.5609 | 349.810 | 0.02455 | 0.453 | 0.526 | 177.00 |
| [MCG+03-08-012](../data/galaxy_sources/mcg_03_08_012.json) | spiral | 108.5821 | 349.878 | 0.02455483 | 1.134 | 0.301 | 52.00 |
| [MCG+08-24-051](../data/galaxy_sources/mcg_08_24_051.json) | spiral | 108.6049 | 349.950 | 0.02456 | 0.529 | 0.698 | 100.00 |
| [Z 35-83](../data/galaxy_sources/z_35_83.json) | spiral | 108.6049 | 349.950 | 0.02456 | 0.811 | 0.986 | 82.00 |
| [MCG+04-26-008](../data/galaxy_sources/mcg_04_26_008.json) | barred spiral | 108.6049 | 349.950 | 0.02456 | 0.531 | 0.817 | 46.00 |
| [Z 46-43](../data/galaxy_sources/z_46_43.json) | weakly barred spiral | 108.6049 | 349.950 | 0.02456 | 0.890 | 0.536 | 24.00 |
| [NGC 840](../data/galaxy_sources/ngc_840.json) | barred spiral | 108.6422 | 350.069 | 0.0245685 | 1.485 | 0.538 | 79.00 |
| [Z 440-13](../data/galaxy_sources/z_440_13.json) | barred spiral | 108.6723 | 350.165 | 0.02457534 | 1.078 | 0.834 | 149.00 |
| [Z 70-14](../data/galaxy_sources/z_70_14.json) | weakly barred spiral | 108.6928 | 350.230 | 0.02458 | 0.768 | 0.611 | 130.00 |
| [NGC 7283](../data/galaxy_sources/ngc_7283.json) | spiral | 108.7174 | 350.308 | 0.0245856 | 0.928 | 0.420 | 17.00 |
| [LEDA 2012141](../data/galaxy_sources/leda_2012141.json) | elliptical | 108.7367 | 350.370 | 0.02459 | 0.388 | 0.555 | 52.00 |
| [Z 379-19](../data/galaxy_sources/z_379_19.json) | spiral | 108.7775 | 350.500 | 0.02459928 | 0.695 | 0.698 | 38.00 |
| [Z 242-63](../data/galaxy_sources/z_242_63.json) | spiral | 108.8247 | 350.650 | 0.02461 | 0.688 | 0.801 | 165.00 |
| [LEDA 2018547](../data/galaxy_sources/leda_2018547.json) | elliptical | 108.8247 | 350.650 | 0.02461 | 0.317 | 0.800 | 133.00 |
| [Z 180-64](../data/galaxy_sources/z_180_64.json) | barred spiral | 108.9565 | 351.070 | 0.02464 | 1.143 | 0.299 | 50.00 |
| [Z 64-29](../data/galaxy_sources/z_64_29.json) | spiral | 108.9565 | 351.070 | 0.02464 | 1.228 | 0.180 | 10.00 |
| [LEDA 37878](../data/galaxy_sources/leda_37878.json) | spiral | 108.9565 | 351.070 | 0.02464 | 0.546 | 0.402 | 75.00 |
| [Z 480-16](../data/galaxy_sources/z_480_16.json) | spiral | 108.9880 | 351.170 | 0.02464716 | 0.969 | 0.203 | 82.00 |
| [Z 405-16](../data/galaxy_sources/z_405_16.json) | spiral | 109.0030 | 351.218 | 0.02465058 | 0.661 | 0.863 | 18.00 |
| [MCG-01-02-032](../data/galaxy_sources/mcg_01_02_032.json) | spiral | 109.0481 | 351.361 | 0.02466084 | 0.945 | 0.341 | 131.00 |
| [UGC 541](../data/galaxy_sources/ugc_541.json) | spiral | 109.0782 | 351.457 | 0.02466768 | 1.198 | 0.181 | 100.00 |
| [Z 382-16](../data/galaxy_sources/z_382_16.json) | spiral | 109.0884 | 351.489 | 0.02467 | 0.814 | 0.703 | 145.00 |
| [Z 46-47](../data/galaxy_sources/z_46_47.json) | spiral | 109.0884 | 351.489 | 0.02467 | 0.620 | 0.358 | 59.00 |
| [Z 239-14](../data/galaxy_sources/z_239_14.json) | spiral | 109.1324 | 351.629 | 0.02468 | 0.615 | 0.468 | 148.00 |
| [LEDA 1356591](../data/galaxy_sources/leda_1356591.json) | barred spiral | 109.1763 | 351.769 | 0.02469 | 0.452 | 0.735 | 79.00 |
| [Z 267-13](../data/galaxy_sources/z_267_13.json) | lenticular | 109.2203 | 351.909 | 0.0247 | 1.021 | 0.455 | 80.00 |
| [MCG+08-23-018](../data/galaxy_sources/mcg_08_23_018.json) | spiral | 109.2203 | 351.909 | 0.0247 | 1.152 | 0.272 | 129.00 |
| [LEDA 1946322](../data/galaxy_sources/leda_1946322.json) | spiral | 109.2203 | 351.909 | 0.0247 | 0.333 | 0.635 | 84.00 |
| [Z 46-48](../data/galaxy_sources/z_46_48.json) | barred spiral | 109.2203 | 351.909 | 0.0247 | 0.808 | 0.859 | 110.00 |
| [Z 208-52](../data/galaxy_sources/z_208_52.json) | spiral | 109.3521 | 352.329 | 0.02473 | 0.728 | 0.318 | 127.00 |
| [MCG+08-24-062](../data/galaxy_sources/mcg_08_24_062.json) | barred spiral | 109.3521 | 352.329 | 0.02473 | 0.507 | 0.862 | 129.00 |
| [UGC 9103](../data/galaxy_sources/ugc_9103.json) | spiral | 109.3521 | 352.329 | 0.02473 | 0.792 | 0.349 | 164.00 |
| [Z 191-71](../data/galaxy_sources/z_191_71.json) | spiral | 109.3521 | 352.329 | 0.02473 | 0.603 | 0.716 | 9.00 |
| [Z 75-7](../data/galaxy_sources/z_75_7.json) | weakly barred spiral | 109.3521 | 352.329 | 0.02473 | 1.090 | 0.240 | 73.00 |
| [Z 180-39](../data/galaxy_sources/z_180_39.json) | spiral | 109.4401 | 352.609 | 0.02475 | 1.002 | 0.301 | 98.00 |
| [Z 72-8](../data/galaxy_sources/z_72_8.json) | spiral | 109.4401 | 352.609 | 0.02475 | 0.686 | 0.624 | 33.00 |
| [Z 475-57](../data/galaxy_sources/z_475_57.json) | spiral | 109.5142 | 352.844 | 0.02476686 | 0.708 | 0.858 | 160.00 |
| [Z 496-39](../data/galaxy_sources/z_496_39.json) | barred spiral | 109.5593 | 352.988 | 0.02477712 | 0.536 | 0.985 | 118.00 |
| [Z 242-29](../data/galaxy_sources/z_242_29.json) | spiral | 109.5719 | 353.028 | 0.02478 | 0.659 | 0.462 | 168.00 |
| [LEDA 213784](../data/galaxy_sources/leda_213784.json) | spiral | 109.5719 | 353.028 | 0.02478 | 0.487 | 0.921 | 47.00 |
| [Z 438-26](../data/galaxy_sources/z_438_26.json) | spiral | 109.5719 | 353.028 | 0.02478 | 0.750 | 0.763 | 21.00 |
| [Z 50-113](../data/galaxy_sources/z_50_113.json) | spiral | 109.5719 | 353.028 | 0.02478 | 0.706 | 0.712 | 135.00 |
| [MCG+09-15-043](../data/galaxy_sources/mcg_09_15_043.json) | spiral | 109.6159 | 353.168 | 0.02479 | 1.174 | 0.166 | 143.00 |
| [UGC 12168](../data/galaxy_sources/ugc_12168.json) | spiral | 109.6194 | 353.179 | 0.0247908 | 0.813 | 0.301 | 35.00 |
| [LEDA 70375](../data/galaxy_sources/leda_70375.json) | spiral | 109.7038 | 353.448 | 0.02481 | 1.069 | 0.226 | 71.00 |
| [ESO 232-23](../data/galaxy_sources/eso_232_23.json) | spiral | 109.7307 | 353.534 | 0.02481612 | 0.140 | 1.000 | 90.00 |
| [Z 241-9](../data/galaxy_sources/z_241_9.json) | spiral | 109.7477 | 353.588 | 0.02482 | 0.802 | 0.598 | 77.00 |
| [LEDA 90909](../data/galaxy_sources/leda_90909.json) | spiral | 109.7477 | 353.588 | 0.02482 | 0.746 | 0.161 | 34.00 |
| [NGC 7353](../data/galaxy_sources/ngc_7353.json) | spiral | 109.7547 | 353.610 | 0.02482159 | 1.510 | 0.400 | 146.00 |
| [Z 265-18](../data/galaxy_sources/z_265_18.json) | elliptical | 109.7917 | 353.728 | 0.02483 | 0.950 | 0.607 | 39.00 |
| [Z 388-7](../data/galaxy_sources/z_388_7.json) | spiral | 109.7917 | 353.728 | 0.02483 | 0.521 | 0.390 | 88.00 |
| [IC 3146](../data/galaxy_sources/ic_3146.json) | spiral | 109.7917 | 353.728 | 0.02483 | 0.390 | 0.849 | 94.00 |
| [Z 440-7](../data/galaxy_sources/z_440_7.json) | spiral | 109.8149 | 353.801 | 0.02483527 | 0.761 | 0.719 | 53.00 |
| [Z 478-45](../data/galaxy_sources/z_478_45.json) | spiral | 109.8299 | 353.849 | 0.02483869 | 0.622 | 0.824 | 123.00 |
| [Z 74-77](../data/galaxy_sources/z_74_77.json) | spiral | 109.8356 | 353.868 | 0.02484 | 0.633 | 0.560 | 128.00 |
| [Z 209-38](../data/galaxy_sources/z_209_38.json) | elliptical | 109.8796 | 354.007 | 0.02485 | 0.800 | 0.868 | 64.00 |
| [Z 243-6](../data/galaxy_sources/z_243_6.json) | elliptical | 109.9236 | 354.147 | 0.02486 | 0.659 | 0.564 | 152.00 |
| [MCG+09-18-051](../data/galaxy_sources/mcg_09_18_051.json) | spiral | 109.9675 | 354.287 | 0.02487 | 0.779 | 0.469 | 42.00 |
| [LEDA 175864](../data/galaxy_sources/leda_175864.json) | elliptical | 109.9675 | 354.287 | 0.02487 | 0.391 | 0.658 | 170.00 |
| [LEDA 70448](../data/galaxy_sources/leda_70448.json) | spiral | 109.9845 | 354.341 | 0.02487387 | 0.406 | 0.828 | 28.00 |
| [Z 405-6](../data/galaxy_sources/z_405_6.json) | spiral | 109.9953 | 354.376 | 0.02487632 | 0.447 | 0.738 | 159.00 |
| [LEDA 3092493](../data/galaxy_sources/leda_3092493.json) | spiral | 110.0115 | 354.427 | 0.02488 | 0.554 | 0.659 | 45.00 |
| [LEDA 2815936](../data/galaxy_sources/leda_2815936.json) | spiral | 110.0554 | 354.567 | 0.02489 | 0.259 | 0.749 | 60.00 |
| [NGC 3677](../data/galaxy_sources/ngc_3677.json) | spiral | 110.0554 | 354.567 | 0.02489 | 1.297 | 0.773 | 130.00 |
| [IC 2254](../data/galaxy_sources/ic_2254.json) | spiral | 110.0554 | 354.567 | 0.02489 | 0.831 | 0.571 | 125.00 |
| [Z 136-95](../data/galaxy_sources/z_136_95.json) | lenticular | 110.0554 | 354.567 | 0.02489 | 0.722 | 0.902 | 0.00 |
| [Z 479-24](../data/galaxy_sources/z_479_24.json) | spiral | 110.1005 | 354.711 | 0.02490026 | 0.585 | 0.598 | 145.00 |
| [Z 241-20](../data/galaxy_sources/z_241_20.json) | barred spiral | 110.1433 | 354.847 | 0.02491 | 1.110 | 0.443 | 176.00 |
| [Z 35-25](../data/galaxy_sources/z_35_25.json) | spiral | 110.1433 | 354.847 | 0.02491 | 0.456 | 0.550 | 46.00 |
| [LEDA 133431](../data/galaxy_sources/leda_133431.json) | lenticular | 110.1873 | 354.987 | 0.02492 | 0.473 | 0.541 | 145.00 |
| [Z 476-34](../data/galaxy_sources/z_476_34.json) | spiral | 110.2058 | 355.045 | 0.02492421 | 0.792 | 0.499 | 50.00 |
| [IC 5145](../data/galaxy_sources/ic_5145.json) | spiral | 110.2208 | 355.093 | 0.02492763 | 2.068 | 0.515 | 169.00 |
| [LEDA 24624](../data/galaxy_sources/leda_24624.json) | spiral | 110.2312 | 355.126 | 0.02493 | 0.369 | 0.569 | 91.00 |
| [Z 263-30](../data/galaxy_sources/z_263_30.json) | spiral | 110.2752 | 355.266 | 0.02494 | 0.914 | 0.959 | 50.00 |
| [Z 501-50](../data/galaxy_sources/z_501_50.json) | spiral | 110.3260 | 355.428 | 0.02495158 | 0.260 | 0.681 | 145.00 |
| [LEDA 32450](../data/galaxy_sources/leda_32450.json) | barred spiral | 110.3631 | 355.546 | 0.02496 | 0.433 | 0.795 | 68.00 |
| [MCG+08-22-011](../data/galaxy_sources/mcg_08_22_011.json) | elliptical | 110.4070 | 355.686 | 0.02497 | 0.566 | 0.854 | 160.00 |
| [LEDA 30769](../data/galaxy_sources/leda_30769.json) | spiral | 110.4070 | 355.686 | 0.02497 | 0.398 | 0.689 | 53.00 |
| [Z 17-70](../data/galaxy_sources/z_17_70.json) | spiral | 110.4070 | 355.686 | 0.02497 | 0.751 | 0.291 | 132.00 |
| [UGC 107](../data/galaxy_sources/ugc_107.json) | spiral | 110.4390 | 355.788 | 0.02497728 | 0.910 | 0.283 | 19.00 |
| [ESO 55-13](../data/galaxy_sources/eso_55_13.json) | lenticular | 110.4418 | 355.797 | 0.02497792 | 0.300 | 0.880 | 35.00 |
| [MCG+04-25-047](../data/galaxy_sources/mcg_04_25_047.json) | spiral | 110.4510 | 355.826 | 0.02498 | 0.740 | 0.869 | 68.00 |
| [Z 213-8](../data/galaxy_sources/z_213_8.json) | lenticular | 110.4510 | 355.826 | 0.02498 | 1.153 | 0.491 | 42.00 |
| [Z 495-9](../data/galaxy_sources/z_495_9.json) | spiral | 110.4614 | 355.859 | 0.02498236 | 0.518 | 0.724 | 22.00 |
| [MCG+09-18-011](../data/galaxy_sources/mcg_09_18_011.json) | spiral | 110.4949 | 355.966 | 0.02499 | 1.293 | 0.130 | 75.00 |
| [Z 499-74](../data/galaxy_sources/z_499_74.json) | spiral | 110.5817 | 356.242 | 0.02500973 | 0.794 | 0.365 | 167.00 |
| [Z 242-27](../data/galaxy_sources/z_242_27.json) | spiral | 110.5828 | 356.245 | 0.02501 | 0.543 | 0.583 | 161.00 |
| [Z 239-40](../data/galaxy_sources/z_239_40.json) | spiral | 110.6268 | 356.385 | 0.02502 | 0.453 | 0.733 | 111.00 |
| [Z 63-94](../data/galaxy_sources/z_63_94.json) | spiral | 110.6268 | 356.385 | 0.02502 | 0.764 | 0.292 | 178.00 |
| [LEDA 87000](../data/galaxy_sources/leda_87000.json) | spiral | 110.6268 | 356.385 | 0.02502 | 0.397 | 0.622 | 57.00 |
| [NGC 4906](../data/galaxy_sources/ngc_4906.json) | elliptical | 110.6268 | 356.385 | 0.02502 | 0.670 | 0.868 | 10.00 |
| [Z 479-32](../data/galaxy_sources/z_479_32.json) | spiral | 110.6418 | 356.433 | 0.02502342 | 0.803 | 0.928 | 92.00 |
| [MCG+08-18-028](../data/galaxy_sources/mcg_08_18_028.json) | spiral | 110.6707 | 356.525 | 0.02503 | 0.732 | 0.566 | 103.00 |
| [LEDA 44723](../data/galaxy_sources/leda_44723.json) | lenticular | 110.6707 | 356.525 | 0.02503 | 0.539 | 0.610 | 141.00 |
| [LEDA 84277](../data/galaxy_sources/leda_84277.json) | spiral | 110.7147 | 356.665 | 0.02504 | 0.479 | 0.950 | 76.00 |
| [IC 3467](../data/galaxy_sources/ic_3467.json) | spiral | 110.7586 | 356.805 | 0.02505 | 1.089 | 0.231 | 71.00 |
| [NGC 6102](../data/galaxy_sources/ngc_6102.json) | spiral | 110.7586 | 356.805 | 0.02505 | 0.789 | 0.736 | 68.00 |
| [LEDA 73275](../data/galaxy_sources/leda_73275.json) | spiral | 110.7922 | 356.911 | 0.02505763 | 0.816 | 0.268 | 129.00 |
| [Z 240-51](../data/galaxy_sources/z_240_51.json) | spiral | 110.8026 | 356.945 | 0.02506 | 0.525 | 0.512 | 73.00 |
| [LEDA 47775](../data/galaxy_sources/leda_47775.json) | spiral | 110.8026 | 356.945 | 0.02506 | 0.560 | 0.588 | 163.00 |
| [MCG+06-30-065](../data/galaxy_sources/mcg_06_30_065.json) | spiral | 110.8026 | 356.945 | 0.02506 | 0.545 | 0.779 | 144.00 |
| [Z 429-10](../data/galaxy_sources/z_429_10.json) | spiral | 110.8222 | 357.007 | 0.02506447 | 0.463 | 0.873 | 164.00 |
| [Z 119-10](../data/galaxy_sources/z_119_10.json) | spiral | 110.8465 | 357.084 | 0.02507 | 0.462 | 0.838 | 171.00 |
| [Z 242-42](../data/galaxy_sources/z_242_42.json) | spiral | 110.8905 | 357.224 | 0.02508 | 1.069 | 0.606 | 61.00 |
| [Z 266-40](../data/galaxy_sources/z_266_40.json) | spiral | 110.9344 | 357.364 | 0.02509 | 0.785 | 0.574 | 99.00 |
| [Z 265-43](../data/galaxy_sources/z_265_43.json) | spiral | 110.9784 | 357.504 | 0.0251 | 0.602 | 0.737 | 78.00 |
| [Z 265-21](../data/galaxy_sources/z_265_21.json) | spiral | 111.0223 | 357.644 | 0.02511 | 0.954 | 0.513 | 170.00 |
| [NGC 5040](../data/galaxy_sources/ngc_5040.json) | elliptical | 111.0663 | 357.784 | 0.02512 | 1.298 | 0.563 | 69.00 |
| [Z 266-36](../data/galaxy_sources/z_266_36.json) | lenticular | 111.1102 | 357.923 | 0.02513 | 0.413 | 0.880 | 31.00 |
| [LEDA 99790](../data/galaxy_sources/leda_99790.json) | elliptical | 111.1102 | 357.923 | 0.02513 | 0.362 | 0.605 | 133.00 |
| [Z 451-15](../data/galaxy_sources/z_451_15.json) | spiral | 111.1230 | 357.964 | 0.0251329 | 0.691 | 0.931 | 120.00 |
| [LEDA 87005](../data/galaxy_sources/leda_87005.json) | spiral | 111.1542 | 358.063 | 0.02514 | 0.280 | 0.600 | 25.00 |
| [MCG+09-18-019](../data/galaxy_sources/mcg_09_18_019.json) | spiral | 111.1981 | 358.203 | 0.02515 | 0.807 | 0.460 | 158.00 |
| [MCG+08-20-084](../data/galaxy_sources/mcg_08_20_084.json) | spiral | 111.2421 | 358.343 | 0.02516 | 1.064 | 0.202 | 15.00 |
| [Z 121-63](../data/galaxy_sources/z_121_63.json) | spiral | 111.2421 | 358.343 | 0.02516 | 1.069 | 0.243 | 77.00 |
| [Z 124-59](../data/galaxy_sources/z_124_59.json) | spiral | 111.2421 | 358.343 | 0.02516 | 0.953 | 0.613 | 158.00 |
| [MCG+09-14-023](../data/galaxy_sources/mcg_09_14_023.json) | barred spiral | 111.2860 | 358.483 | 0.02517 | 0.691 | 0.493 | 6.00 |
| [Z 150-10](../data/galaxy_sources/z_150_10.json) | spiral | 111.2860 | 358.483 | 0.02517 | 0.702 | 0.538 | 111.00 |
| [IC 544](../data/galaxy_sources/ic_544.json) | spiral | 111.2860 | 358.483 | 0.02517 | 0.921 | 0.303 | 24.00 |
| [Z 47-121](../data/galaxy_sources/z_47_121.json) | spiral | 111.2860 | 358.483 | 0.02517 | 0.714 | 0.432 | 98.00 |
| [Z 390-82](../data/galaxy_sources/z_390_82.json) | spiral | 111.3299 | 358.622 | 0.02518 | 1.226 | 0.198 | 61.00 |
| [Z 211-28](../data/galaxy_sources/z_211_28.json) | lenticular | 111.3299 | 358.622 | 0.02518 | 0.809 | 0.683 | 24.00 |
| [IC 3170](../data/galaxy_sources/ic_3170.json) | spiral | 111.3299 | 358.622 | 0.02518 | 0.575 | 0.980 | 115.00 |
| [LEDA 25501](../data/galaxy_sources/leda_25501.json) | spiral | 111.3739 | 358.762 | 0.02519 | 0.317 | 0.842 | 45.00 |
| [Z 220-31](../data/galaxy_sources/z_220_31.json) | weakly barred spiral | 111.4178 | 358.902 | 0.0252 | 1.066 | 0.575 | 139.00 |
| [Z 187-39](../data/galaxy_sources/z_187_39.json) | spiral | 111.4178 | 358.902 | 0.0252 | 0.818 | 0.354 | 55.00 |
| [UGC 12072](../data/galaxy_sources/ugc_12072.json) | spiral | 111.4237 | 358.921 | 0.02520134 | 0.891 | 0.231 | 44.00 |
| [LEDA 74041](../data/galaxy_sources/leda_74041.json) | spiral | 111.4538 | 359.016 | 0.02520818 | 0.399 | 0.607 | 3.00 |
| [IC 619](../data/galaxy_sources/ic_619.json) | spiral | 111.4618 | 359.042 | 0.02521 | 0.979 | 0.695 | 175.00 |
| [Z 440-9](../data/galaxy_sources/z_440_9.json) | spiral | 111.4688 | 359.064 | 0.0252116 | 0.600 | 0.808 | 86.00 |
| [Z 290-31](../data/galaxy_sources/z_290_31.json) | spiral | 111.5057 | 359.182 | 0.02522 | 0.742 | 0.672 | 111.00 |
| [Z 124-55](../data/galaxy_sources/z_124_55.json) | barred spiral | 111.5139 | 359.208 | 0.02522187 | 1.148 | 0.500 | 175.00 |
| [LEDA 82460](../data/galaxy_sources/leda_82460.json) | spiral | 111.5497 | 359.322 | 0.02523 | 0.596 | 0.538 | 2.00 |
| [Z 11-20](../data/galaxy_sources/z_11_20.json) | spiral | 111.5591 | 359.351 | 0.02523213 | 1.273 | 0.152 | 156.00 |
| [Z 212-55](../data/galaxy_sources/z_212_55.json) | lenticular | 111.5936 | 359.461 | 0.02524 | 0.491 | 0.465 | 132.00 |
| [UGC 12197](../data/galaxy_sources/ugc_12197.json) | spiral | 111.6342 | 359.591 | 0.02524924 | 1.112 | 0.168 | 117.00 |
| [Z 106-44](../data/galaxy_sources/z_106_44.json) | spiral | 111.6376 | 359.601 | 0.02525 | 0.870 | 0.394 | 107.00 |
| [Z 240-39](../data/galaxy_sources/z_240_39.json) | spiral | 111.6815 | 359.741 | 0.02526 | 0.454 | 0.713 | 71.00 |
| [LEDA 1816756](../data/galaxy_sources/leda_1816756.json) | spiral | 111.6815 | 359.741 | 0.02526 | 0.392 | 0.956 | 66.00 |
| [LEDA 1149944](../data/galaxy_sources/leda_1149944.json) | spiral | 111.6815 | 359.741 | 0.02526 | 1.070 | 0.186 | 104.00 |
| [Z 498-66](../data/galaxy_sources/z_498_66.json) | spiral | 111.6944 | 359.782 | 0.02526293 | 0.511 | 0.715 | 16.00 |
| [IC 2369](../data/galaxy_sources/ic_2369.json) | spiral | 111.7255 | 359.881 | 0.02527 | 1.350 | 0.221 | 86.00 |
| [ESO 429-14](../data/galaxy_sources/eso_429_14.json) | spiral | 111.7545 | 359.973 | 0.02527662 | 0.830 | 0.200 | 80.00 |
| [Z 102-44](../data/galaxy_sources/z_102_44.json) | spiral | 111.7545 | 359.973 | 0.02527662 | 0.593 | 0.453 | 53.00 |
| [Z 39-37](../data/galaxy_sources/z_39_37.json) | spiral | 111.7694 | 360.020 | 0.02528 | 0.530 | 0.809 | 164.00 |
| [MCG+09-22-065](../data/galaxy_sources/mcg_09_22_065.json) | spiral | 111.8133 | 360.160 | 0.02529 | 0.713 | 0.626 | 106.00 |
| [LEDA 33050](../data/galaxy_sources/leda_33050.json) | spiral | 111.8133 | 360.160 | 0.02529 | 0.343 | 0.662 | 152.00 |
| [MCG+02-58-048](../data/galaxy_sources/mcg_02_58_048.json) | spiral | 111.8133 | 360.160 | 0.02529 | 0.936 | 0.393 | 37.00 |
| [LEDA 85263](../data/galaxy_sources/leda_85263.json) | spiral | 111.8147 | 360.165 | 0.02529031 | 0.444 | 0.676 | 21.00 |
| [Z 265-15](../data/galaxy_sources/z_265_15.json) | spiral | 111.8573 | 360.300 | 0.0253 | 0.769 | 0.265 | 162.00 |
| [Z 240-14](../data/galaxy_sources/z_240_14.json) | spiral | 111.8573 | 360.300 | 0.0253 | 0.830 | 0.314 | 2.00 |
| [Z 11-93](../data/galaxy_sources/z_11_93.json) | elliptical | 111.8573 | 360.300 | 0.0253 | 1.212 | 0.849 | 107.00 |
| [Z 122-49](../data/galaxy_sources/z_122_49.json) | spiral | 111.9452 | 360.580 | 0.02532 | 1.259 | 0.284 | 106.00 |
| [LEDA 3092588](../data/galaxy_sources/leda_3092588.json) | lenticular | 111.9452 | 360.580 | 0.02532 | 0.388 | 0.790 | 163.00 |
| [NGC 7348](../data/galaxy_sources/ngc_7348.json) | barred spiral | 112.0102 | 360.786 | 0.0253348 | 1.142 | 0.541 | 10.00 |
| [LEDA 44792](../data/galaxy_sources/leda_44792.json) | barred lenticular | 112.0770 | 360.999 | 0.02535 | 0.630 | 0.760 | 86.00 |
| [Z 512-1](../data/galaxy_sources/z_512_1.json) | spiral | 112.0854 | 361.026 | 0.02535191 | 0.957 | 0.280 | 160.00 |
| [MCG+09-15-068](../data/galaxy_sources/mcg_09_15_068.json) | elliptical | 112.1210 | 361.139 | 0.02536 | 0.650 | 0.510 | 31.00 |
| [Z 212-49](../data/galaxy_sources/z_212_49.json) | elliptical | 112.1210 | 361.139 | 0.02536 | 0.915 | 0.684 | 8.00 |
| [Z 102-63](../data/galaxy_sources/z_102_63.json) | spiral | 112.1210 | 361.139 | 0.02536 | 0.566 | 0.889 | 164.00 |
| [LEDA 1177547](../data/galaxy_sources/leda_1177547.json) | spiral | 112.1210 | 361.139 | 0.02536 | 0.650 | 0.241 | 143.00 |
| [Z 213-26](../data/galaxy_sources/z_213_26.json) | lenticular | 112.1649 | 361.279 | 0.02537 | 1.163 | 0.638 | 77.00 |
| [Z 266-57](../data/galaxy_sources/z_266_57.json) | spiral | 112.2088 | 361.418 | 0.02538 | 0.693 | 0.650 | 77.00 |
| [Z 5-4](../data/galaxy_sources/z_5_4.json) | spiral | 112.2088 | 361.418 | 0.02538 | 0.496 | 0.781 | 32.00 |
| [IC 1839](../data/galaxy_sources/ic_1839.json) | spiral | 112.2508 | 361.552 | 0.02538956 | 1.203 | 0.426 | 96.00 |
| [Z 288-15](../data/galaxy_sources/z_288_15.json) | lenticular | 112.2528 | 361.558 | 0.02539 | 0.477 | 0.844 | 171.00 |
| [Z 266-16](../data/galaxy_sources/z_266_16.json) | spiral | 112.2528 | 361.558 | 0.02539 | 0.747 | 0.520 | 179.00 |
| [LEDA 84422](../data/galaxy_sources/leda_84422.json) | spiral | 112.3407 | 361.838 | 0.02541 | 0.643 | 0.939 | 168.00 |
| [IC 2230](../data/galaxy_sources/ic_2230.json) | lenticular | 112.3407 | 361.838 | 0.02541 | 0.633 | 0.740 | 164.00 |
| [UGC 189](../data/galaxy_sources/ugc_189.json) | barred spiral | 112.3407 | 361.838 | 0.02541 | 0.603 | 0.656 | 119.00 |
| [Z 43-55](../data/galaxy_sources/z_43_55.json) | barred spiral | 112.3407 | 361.838 | 0.02541 | 0.656 | 0.683 | 129.00 |
| [Z 440-14](../data/galaxy_sources/z_440_14.json) | spiral | 112.3561 | 361.887 | 0.02541351 | 0.787 | 0.351 | 38.00 |
| [Z 151-67](../data/galaxy_sources/z_151_67.json) | spiral | 112.3846 | 361.977 | 0.02542 | 0.577 | 0.894 | 74.00 |
| [Z 34-23](../data/galaxy_sources/z_34_23.json) | spiral | 112.3846 | 361.977 | 0.02542 | 0.428 | 0.905 | 54.00 |
| [Z 208-23](../data/galaxy_sources/z_208_23.json) | lenticular | 112.4285 | 362.117 | 0.02543 | 0.755 | 0.792 | 117.00 |
| [LEDA 82368](../data/galaxy_sources/leda_82368.json) | spiral | 112.4285 | 362.117 | 0.02543 | 0.193 | 0.824 | 10.00 |
| [Z 503-39](../data/galaxy_sources/z_503_39.json) | spiral | 112.4463 | 362.174 | 0.02543405 | 0.644 | 0.531 | 157.00 |
| [IC 4394](../data/galaxy_sources/ic_4394.json) | spiral | 112.4725 | 362.257 | 0.02544 | 0.834 | 0.466 | 14.00 |
| [MCG+03-27-009](../data/galaxy_sources/mcg_03_27_009.json) | lenticular | 112.5164 | 362.397 | 0.02545 | 0.564 | 0.625 | 164.00 |
| [Z 379-28](../data/galaxy_sources/z_379_28.json) | weakly barred spiral | 112.5275 | 362.432 | 0.02545253 | 0.717 | 0.838 | 27.00 |
| [2MASX J12012005-5347555](../data/galaxy_sources/galaxy_2masx_j12012005_5347555.json) | barred spiral | 112.5306 | 362.442 | 0.02545322 | 0.350 | 0.880 | 25.00 |
| [Z 238-62](../data/galaxy_sources/z_238_62.json) | spiral | 112.5604 | 362.536 | 0.02546 | 1.024 | 0.469 | 49.00 |
| [MCG+07-23-017](../data/galaxy_sources/mcg_07_23_017.json) | spiral | 112.5604 | 362.536 | 0.02546 | 0.812 | 0.417 | 118.00 |
| [Z 454-31](../data/galaxy_sources/z_454_31.json) | spiral | 112.6268 | 362.748 | 0.02547512 | 0.960 | 0.334 | 49.00 |
| [Z 272-10](../data/galaxy_sources/z_272_10.json) | spiral | 112.6922 | 362.956 | 0.02549 | 0.648 | 0.750 | 38.00 |
| [Z 120-28](../data/galaxy_sources/z_120_28.json) | barred spiral | 112.6922 | 362.956 | 0.02549 | 0.538 | 0.736 | 3.00 |
| [Z 477-49](../data/galaxy_sources/z_477_49.json) | spiral | 112.7020 | 362.987 | 0.02549224 | 0.620 | 0.884 | 71.00 |
| [Z 149-47](../data/galaxy_sources/z_149_47.json) | spiral | 112.7361 | 363.095 | 0.0255 | 0.631 | 0.412 | 165.00 |
| [Z 198-21](../data/galaxy_sources/z_198_21.json) | spiral | 112.7361 | 363.095 | 0.0255 | 1.166 | 0.251 | 100.00 |
| [Z 474-4](../data/galaxy_sources/z_474_4.json) | barred spiral | 112.7772 | 363.226 | 0.02550935 | 0.960 | 0.828 | 109.00 |
| [Z 184-58](../data/galaxy_sources/z_184_58.json) | spiral | 112.7801 | 363.235 | 0.02551 | 0.573 | 0.625 | 2.00 |
| [MCG-06-13-018](../data/galaxy_sources/mcg_06_13_018.json) | spiral | 112.7802 | 363.236 | 0.02551003 | 0.896 | 0.683 | 35.00 |
| [Z 477-53](../data/galaxy_sources/z_477_53.json) | spiral | 112.7922 | 363.274 | 0.02551277 | 1.200 | 0.341 | 10.00 |
| [UGC 12249](../data/galaxy_sources/ugc_12249.json) | spiral | 112.8374 | 363.417 | 0.02552304 | 1.296 | 0.154 | 54.00 |
| [LEDA 86403](../data/galaxy_sources/leda_86403.json) | spiral | 112.8679 | 363.515 | 0.02553 | 0.691 | 0.294 | 88.00 |
| [Z 290-20](../data/galaxy_sources/z_290_20.json) | spiral | 112.9119 | 363.654 | 0.02554 | 1.145 | 0.427 | 96.00 |
| [Z 372-4](../data/galaxy_sources/z_372_4.json) | spiral | 112.9577 | 363.800 | 0.02555043 | 0.133 | 1.000 | 90.00 |
| [Z 405-4](../data/galaxy_sources/z_405_4.json) | spiral | 112.9727 | 363.848 | 0.02555385 | 1.429 | 0.469 | 93.00 |
| [LEDA 7974](../data/galaxy_sources/leda_7974.json) | spiral | 113.0179 | 363.991 | 0.02556412 | 0.584 | 0.326 | 159.00 |
| [MCG+09-23-013](../data/galaxy_sources/mcg_09_23_013.json) | spiral | 113.0437 | 364.074 | 0.02557 | 0.709 | 0.324 | 63.00 |
| [NGC 5497](../data/galaxy_sources/ngc_5497.json) | spiral | 113.0437 | 364.074 | 0.02557 | 0.805 | 0.702 | 59.00 |
| [ESO 327-32](../data/galaxy_sources/eso_327_32.json) | barred spiral | 113.0570 | 364.116 | 0.02557302 | 0.583 | 0.501 | 145.00 |
| [Z 409-7](../data/galaxy_sources/z_409_7.json) | spiral | 113.0780 | 364.183 | 0.02557781 | 0.911 | 0.322 | 51.00 |
| [IC 990](../data/galaxy_sources/ic_990.json) | elliptical | 113.0876 | 364.213 | 0.02558 | 0.760 | 0.766 | 64.00 |
| [Z 72-31](../data/galaxy_sources/z_72_31.json) | elliptical | 113.0876 | 364.213 | 0.02558 | 0.522 | 0.889 | 62.00 |
| [Z 122-15](../data/galaxy_sources/z_122_15.json) | spiral | 113.1316 | 364.353 | 0.02559 | 0.764 | 0.822 | 25.00 |
| [Z 11-91](../data/galaxy_sources/z_11_91.json) | elliptical | 113.1316 | 364.353 | 0.02559 | 0.776 | 0.704 | 121.00 |
| [Z 121-22](../data/galaxy_sources/z_121_22.json) | spiral | 113.2195 | 364.633 | 0.02561 | 1.137 | 0.199 | 43.00 |
| [Z 213-20](../data/galaxy_sources/z_213_20.json) | spiral | 113.2195 | 364.633 | 0.02561 | 0.519 | 0.677 | 48.00 |
| [Z 67-11](../data/galaxy_sources/z_67_11.json) | spiral | 113.2195 | 364.633 | 0.02561 | 0.680 | 0.725 | 57.00 |
| [ZW VIII 275](../data/galaxy_sources/zw_viii_275.json) | spiral | 113.2634 | 364.772 | 0.02562 | 0.462 | 0.646 | 164.00 |
| [Z 212-28](../data/galaxy_sources/z_212_28.json) | elliptical | 113.3073 | 364.912 | 0.02563 | 1.140 | 0.761 | 26.00 |
| [LEDA 156664](../data/galaxy_sources/leda_156664.json) | spiral | 113.3073 | 364.912 | 0.02563 | 0.594 | 0.297 | 8.00 |
| [Z 103-75](../data/galaxy_sources/z_103_75.json) | spiral | 113.3073 | 364.912 | 0.02563 | 0.547 | 0.657 | 114.00 |
| [Z 77-12](../data/galaxy_sources/z_77_12.json) | spiral | 113.3073 | 364.912 | 0.02563 | 0.888 | 0.336 | 86.00 |
| [Z 179-23](../data/galaxy_sources/z_179_23.json) | weakly barred spiral | 113.3513 | 365.052 | 0.02564 | 1.505 | 0.205 | 16.00 |
| [Z 120-39](../data/galaxy_sources/z_120_39.json) | barred spiral | 113.3513 | 365.052 | 0.02564 | 0.505 | 0.793 | 136.00 |
| [Z 184-35](../data/galaxy_sources/z_184_35.json) | barred spiral | 113.3513 | 365.052 | 0.02564 | 0.831 | 0.770 | 81.00 |
| [Z 473-13](../data/galaxy_sources/z_473_13.json) | spiral | 113.3638 | 365.092 | 0.02564285 | 0.611 | 0.517 | 90.00 |
| [MCG+09-17-038](../data/galaxy_sources/mcg_09_17_038.json) | spiral | 113.3952 | 365.191 | 0.02565 | 0.865 | 0.247 | 116.00 |
| [IC 2387](../data/galaxy_sources/ic_2387.json) | spiral | 113.3952 | 365.191 | 0.02565 | 1.430 | 0.406 | 18.00 |
| [Z 198-19](../data/galaxy_sources/z_198_19.json) | spiral | 113.3952 | 365.191 | 0.02565 | 0.605 | 0.701 | 60.00 |
| [Z 34-11](../data/galaxy_sources/z_34_11.json) | barred spiral | 113.4391 | 365.331 | 0.02566 | 0.647 | 0.667 | 89.00 |
| [LEDA 99789](../data/galaxy_sources/leda_99789.json) | spiral | 113.4391 | 365.331 | 0.02566 | 0.514 | 0.374 | 142.00 |
| [LEDA 24207](../data/galaxy_sources/leda_24207.json) | spiral | 113.4831 | 365.471 | 0.02567 | 0.525 | 0.399 | 124.00 |
| [Z 120-45](../data/galaxy_sources/z_120_45.json) | spiral | 113.4831 | 365.471 | 0.02567 | 1.186 | 0.880 | 86.00 |
| [IC 1011](../data/galaxy_sources/ic_1011.json) | spiral | 113.4831 | 365.471 | 0.02567 | 0.629 | 0.681 | 115.00 |
| [Z 478-41](../data/galaxy_sources/z_478_41.json) | spiral | 113.4992 | 365.522 | 0.02567366 | 1.132 | 0.417 | 106.00 |
| [Z 263-36](../data/galaxy_sources/z_263_36.json) | spiral | 113.5270 | 365.611 | 0.02568 | 1.186 | 0.441 | 4.00 |
| [NGC 5541](../data/galaxy_sources/ngc_5541.json) | spiral | 113.5270 | 365.611 | 0.02568 | 1.020 | 0.703 | 26.00 |
| [Z 75-80](../data/galaxy_sources/z_75_80.json) | lenticular | 113.5270 | 365.611 | 0.02568 | 0.496 | 0.779 | 78.00 |
| [Z 47-41](../data/galaxy_sources/z_47_41.json) | spiral | 113.5270 | 365.611 | 0.02568 | 1.021 | 0.405 | 153.00 |
| [LEDA 173206](../data/galaxy_sources/leda_173206.json) | lenticular | 113.6149 | 365.890 | 0.0257 | 0.708 | 0.468 | 33.00 |
| [Z 372-1](../data/galaxy_sources/z_372_1.json) | barred spiral | 113.6496 | 366.000 | 0.0257079 | 0.580 | 0.600 | 5.00 |
| [Z 430-43](../data/galaxy_sources/z_430_43.json) | spiral | 113.6588 | 366.030 | 0.02571 | 0.850 | 0.399 | 65.00 |
| [Z 454-19](../data/galaxy_sources/z_454_19.json) | spiral | 113.6646 | 366.048 | 0.02571132 | 0.947 | 0.355 | 35.00 |
| [Z 267-7](../data/galaxy_sources/z_267_7.json) | lenticular | 113.7027 | 366.169 | 0.02572 | 0.451 | 0.766 | 40.00 |
| [LEDA 33263](../data/galaxy_sources/leda_33263.json) | spiral | 113.7027 | 366.169 | 0.02572 | 0.345 | 0.742 | 125.00 |
| [Z 413-32](../data/galaxy_sources/z_413_32.json) | spiral | 113.7398 | 366.287 | 0.02572844 | 0.915 | 0.420 | 37.00 |
| [Z 387-69](../data/galaxy_sources/z_387_69.json) | spiral | 113.7467 | 366.309 | 0.02573 | 0.530 | 0.595 | 58.00 |
| [LEDA 87074](../data/galaxy_sources/leda_87074.json) | spiral | 113.7467 | 366.309 | 0.02573 | 0.334 | 0.810 | 72.00 |
| [MCG+08-27-010](../data/galaxy_sources/mcg_08_27_010.json) | spiral | 113.7906 | 366.449 | 0.02574 | 0.763 | 0.371 | 110.00 |
| [LEDA 82359](../data/galaxy_sources/leda_82359.json) | spiral | 113.7906 | 366.449 | 0.02574 | 0.399 | 0.745 | 116.00 |
| [Z 18-83](../data/galaxy_sources/z_18_83.json) | spiral | 113.7906 | 366.449 | 0.02574 | 0.638 | 0.309 | 36.00 |
| [MCG+09-16-020](../data/galaxy_sources/mcg_09_16_020.json) | spiral | 113.8346 | 366.589 | 0.02575 | 0.841 | 0.221 | 81.00 |
| [MCG+07-28-033](../data/galaxy_sources/mcg_07_28_033.json) | spiral | 113.8346 | 366.589 | 0.02575 | 0.890 | 0.276 | 133.00 |
| [ZW VIII 444](../data/galaxy_sources/zw_viii_444.json) | lenticular | 113.8346 | 366.589 | 0.02575 | 0.326 | 0.902 | 161.00 |
| [Z 51-9](../data/galaxy_sources/z_51_9.json) | barred spiral | 113.8346 | 366.589 | 0.02575 | 0.703 | 0.888 | 31.00 |
| [Z 477-42](../data/galaxy_sources/z_477_42.json) | spiral | 113.8602 | 366.670 | 0.02575583 | 1.098 | 0.234 | 49.00 |
| [Z 92-33](../data/galaxy_sources/z_92_33.json) | elliptical | 113.9224 | 366.868 | 0.02577 | 0.730 | 0.629 | 4.00 |
| [Z 12-12](../data/galaxy_sources/z_12_12.json) | spiral | 113.9224 | 366.868 | 0.02577 | 0.502 | 0.432 | 69.00 |
| [IC 3127](../data/galaxy_sources/ic_3127.json) | spiral | 113.9224 | 366.868 | 0.02577 | 0.602 | 0.840 | 180.00 |
| [Z 212-47](../data/galaxy_sources/z_212_47.json) | spiral | 113.9664 | 367.008 | 0.02578 | 0.598 | 0.633 | 49.00 |
| [Z 96-8](../data/galaxy_sources/z_96_8.json) | lenticular | 113.9664 | 367.008 | 0.02578 | 0.504 | 0.885 | 130.00 |
| [Z 73-34](../data/galaxy_sources/z_73_34.json) | spiral | 114.0103 | 367.147 | 0.02579 | 0.650 | 0.538 | 141.00 |
| [NGC 3214](../data/galaxy_sources/ngc_3214.json) | lenticular | 114.0982 | 367.427 | 0.02581 | 0.961 | 0.576 | 40.00 |
| [IC 2475](../data/galaxy_sources/ic_2475.json) | spiral | 114.0982 | 367.427 | 0.02581 | 1.024 | 0.374 | 107.00 |
| [Z 18-66](../data/galaxy_sources/z_18_66.json) | spiral | 114.0982 | 367.427 | 0.02581 | 0.667 | 0.325 | 143.00 |
| [MCG+09-14-051](../data/galaxy_sources/mcg_09_14_051.json) | spiral | 114.1421 | 367.566 | 0.02582 | 0.513 | 0.638 | 156.00 |
| [Z 120-52](../data/galaxy_sources/z_120_52.json) | spiral | 114.1421 | 367.566 | 0.02582 | 1.376 | 0.153 | 77.00 |
