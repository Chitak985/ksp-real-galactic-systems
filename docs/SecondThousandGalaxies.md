# The second thousand generated galaxies

This batch adds **1,000** real SIMBAD galaxies, bringing the generated catalog
to **2,000** (2,003 GalacticDisk definitions including the three legacy galaxy
bodies). All previous adopted objects, configs, visual refinements, source
records and the original source snapshot are preserved. Sun and README are
unchanged; no commits are made.

## Sources and selection

The previous source snapshot had only 959 further eligible objects. A bounded
continuation query at the [SIMBAD South African mirror](https://simbad.idia.ac.za/simbad/) supplied
4,000 candidate records. It uses the previous fields and non-null requirements,
but begins at z=0.02582, the last accepted redshift of the preceding batch, and
retains the z<=0.1 ceiling. Ordering remains spectroscopic redshift, then SIMBAD
object ID. The cutoff is a reproducible continuation of this selection, not a
claim that no closer unmodeled galaxies exist. Mirror contents can differ in
update date; original measurements and their references are retained.

The exact ADQL query, URL, retrieval timestamp, original response text, parsed
rows and SHA256 are saved in `data/galaxy_sources/simbad_bulk_continuation_snapshot.json`.
The 1,000 accepted IDs and reasons for 1,290 skipped candidates encountered before
the final accepted object are saved in `data/bulk_galaxy_selection.json`.
The previous 900-galaxy audit remains in `previous_selections`, and its original
`simbad_bulk_snapshot.json` is unchanged.

Each new raw record points to an immutable snapshot file named with the full
SHA256 of its original response. Later imports can replace the latest-snapshot
alias while preserving these files and previous audits. The importer checks
both the response checksum and equality of the parsed rows with that response.
These checksums establish internal integrity, not publisher-signed authenticity.

This remains a heterogeneous literature compilation, not a complete or unbiased
galaxy survey. Original morphology, dimensions and redshift bibcodes are saved
per object; B/K fluxes are recorded as SIMBAD photometry. See the
[SIMBAD database paper](https://arxiv.org/abs/astro-ph/0002110) and preserve
SIMBAD's ODbL attribution when redistributing this data compilation.

## Quality and modeling

The same acceptance gates as the [preceding 900-galaxy batch](BulkGalaxies.md)
remain in place: explicit spectroscopic redshifts; A/B/C redshift and dimension
quality; positive measured axes and a sky position angle; B/K photometry;
original morphology/dimension/redshift references; and a supported regular
shape without ambiguity markers. Implausible inferred sizes, stellar masses,
extreme flattening, existing aliases and objects within five arcseconds of
another accepted/existing galaxy are skipped for review. No quality threshold
was relaxed to reach 1,000. No new irregular, ring or merging template is used.

The additions span z=0.02583-0.03061942, approximately **372.4-441.0 million
light-years in comoving placement distance**, with lookback times of about
**367.7-434.4 million years**. Comoving distance divided by c is not the lookback
time. The existing flat FLRW model uses H0=67.4 km/s/Mpc, Omega_m=0.315,
Omega_r=0.000092; angular-diameter distance preserves observed angular sizes,
and luminosity distance is used for the Ks stellar-mass estimate.

Descriptions retain explicit uncertainties: peculiar velocities affect distance,
Ks mass/light=1 is an estimate excluding the dark halo, and isophotal size,
3D depth, colors, spiral phase and core gravity are approximate. Measured PA and
apparent flattening determine observed orientation; detailed structure, disk
inclination and near-side choice remain model assumptions. Observed light already
comes from the emission epoch, so morphology is not rewound again.

Every new barycenter is a file directly under `RealGalacticSystems/Systems`.
Each has the existing hidden circular Sun-parented position-only orbit,
`period = 3E300` and `mode = OFF`, static disk rotation, and a small 5-100 pc core
SOI. A pairwise check found no overlapping core SOIs involving the additions
among all 2,000 generated galaxy barycenters.

Each new object retains the **2,000 representative mesh-star** rendering budget,
adding two million mesh stars. This is a display sample, not a physical stellar
population. Diffuse volumes remain enabled. Existing objects and template
defaults are unchanged. Live KSP startup, memory and frame rate at 2,000 galaxies
remain unmeasured; no distance LOD or simultaneous game-load validation is claimed.

## Composition

| Family | Added |
|---|---:|
| Spiral | 709 |
| Lenticular | 92 |
| Weakly barred spiral | 12 |
| Barred spiral | 95 |
| Elliptical | 90 |
| Barred lenticular | 2 |

## Rebuilding and further imports

```powershell
python rebuild_galaxies.py
python rebuild_galaxies.py --check
python fetch_bulk_galaxies.py --offline --count 50 --dry-run
```

The importer defaults to a requested count of 1,000. `--offline` uses the saved
continuation sample; without it, the same bounded continuation query is fetched
from SIMBAD with mirror fallback. The query remains capped at 4,000 candidate
rows; this does not promise 4,000 eligible additions. Exhaustion fails before
writes. Legacy snapshots remain accepted for verification, and every new import
archives its response by checksum before it can be superseded by a later one.

Repeated cosmological integrals are now cached in memory during a run, keyed by
redshift and all cosmological settings. Inputs are still validated on every call
and each caller gets a fresh result dictionary. The cache writes no files and
does not change calculations or generated configs; its purpose is to avoid
recomputing the same distances repeatedly while rebuilding thousands of objects.

## Verification

All 62 automated tests pass, including original-source replay, quality and
morphology filters, duplicate protection, angular-size and sky-orientation
projection, frozen placement, small core SOIs, template propagation, offline
dry runs, preserved source histories and cosmology-cache isolation. The source
history test also passes with parsed snapshots reused during the test rather
than repeatedly reading the same multi-megabyte response for every object.
All 2,000 generated galaxy configs and 150 generated star configs match their
saved data and templates. Orientation checks cover all 2,003 disk definitions.

Nineteen representative additions cover all six families and their extremes of
size and flattening. They were rendered and visually inspected using the current
GalacticDisk C# renderer and its packaged shader in isolated Unity 2019.4, with
enlarged views along the Sol-to-galaxy direction. No individual image match or
absolute photometric calibration is claimed. These previews use KSP API
substitutes, do not open a user save, and do not measure the cost of loading all
2,000 galaxies together. The original 1,000 galaxy configs, legacy galaxy configs,
all star configs and original source files were compared byte for byte.

## Full addition list

Values are nominal, not precision guarantees. PA is north through east modulo
180 degrees; q is the apparent minor/major axis ratio. Source-record links retain
the measurement references and modeling assumptions.

| Galaxy / saved source | Family | Comoving placement (Mpc) | Lookback (Myr) | z | Major axis (arcmin) | q | PA (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| [IC 1775](../data/galaxy_sources/ic_1775.json) | spiral | 114.1860 | 367.706 | 0.02583 | 0.720 | 0.911 | 159.00 |
| [Z 430-66](../data/galaxy_sources/z_430_66.json) | spiral | 114.3114 | 368.105 | 0.02585855 | 0.871 | 0.711 | 10.00 |
| [LEDA 104564](../data/galaxy_sources/leda_104564.json) | spiral | 114.3617 | 368.265 | 0.02587 | 0.475 | 0.291 | 142.00 |
| [Z 19-14](../data/galaxy_sources/z_19_14.json) | spiral | 114.3617 | 368.265 | 0.02587 | 0.624 | 0.450 | 20.00 |
| [NGC 3385](../data/galaxy_sources/ngc_3385.json) | lenticular | 114.4057 | 368.404 | 0.02588 | 1.908 | 0.528 | 96.00 |
| [LEDA 53572](../data/galaxy_sources/leda_53572.json) | spiral | 114.4496 | 368.544 | 0.02589 | 1.042 | 0.204 | 157.00 |
| [Z 288-29](../data/galaxy_sources/z_288_29.json) | spiral | 114.4496 | 368.544 | 0.02589 | 0.803 | 0.973 | 140.00 |
| [LEDA 70050](../data/galaxy_sources/leda_70050.json) | spiral | 114.4496 | 368.544 | 0.02589 | 0.321 | 0.901 | 0.00 |
| [MCG-02-03-038](../data/galaxy_sources/mcg_02_03_038.json) | spiral | 114.4935 | 368.684 | 0.0259 | 0.560 | 0.876 | 137.00 |
| [LEDA 176224](../data/galaxy_sources/leda_176224.json) | spiral | 114.4935 | 368.684 | 0.0259 | 0.517 | 0.616 | 70.00 |
| [Z 503-47](../data/galaxy_sources/z_503_47.json) | weakly barred spiral | 114.5070 | 368.727 | 0.02590306 | 0.824 | 0.923 | 167.00 |
| [MCG-06-32-005](../data/galaxy_sources/mcg_06_32_005.json) | spiral | 114.5702 | 368.927 | 0.02591744 | 0.697 | 0.839 | 70.00 |
| [Z 265-9](../data/galaxy_sources/z_265_9.json) | barred spiral | 114.5814 | 368.963 | 0.02592 | 0.741 | 0.706 | 22.00 |
| [MCG+09-14-026](../data/galaxy_sources/mcg_09_14_026.json) | spiral | 114.6253 | 369.103 | 0.02593 | 1.145 | 0.221 | 77.00 |
| [LEDA 57425](../data/galaxy_sources/leda_57425.json) | spiral | 114.6253 | 369.103 | 0.02593 | 0.417 | 0.474 | 139.00 |
| [IC 992](../data/galaxy_sources/ic_992.json) | spiral | 114.6693 | 369.242 | 0.02594 | 0.833 | 0.803 | 107.00 |
| [Z 461-48](../data/galaxy_sources/z_461_48.json) | spiral | 114.7026 | 369.348 | 0.02594758 | 1.060 | 0.425 | 168.00 |
| [Z 126-20](../data/galaxy_sources/z_126_20.json) | spiral | 114.7132 | 369.382 | 0.02595 | 0.823 | 0.842 | 101.00 |
| [Z 103-104](../data/galaxy_sources/z_103_104.json) | spiral | 114.7132 | 369.382 | 0.02595 | 0.475 | 0.618 | 35.00 |
| [MCG+07-29-061](../data/galaxy_sources/mcg_07_29_061.json) | spiral | 114.7132 | 369.382 | 0.02595 | 0.596 | 0.957 | 19.00 |
| [Z 243-5](../data/galaxy_sources/z_243_5.json) | spiral | 114.7571 | 369.522 | 0.02596 | 0.480 | 0.599 | 91.00 |
| [Z 102-21](../data/galaxy_sources/z_102_21.json) | spiral | 114.7571 | 369.522 | 0.02596 | 1.216 | 0.221 | 154.00 |
| [Z 136-97](../data/galaxy_sources/z_136_97.json) | spiral | 114.8011 | 369.661 | 0.02597 | 1.097 | 0.529 | 160.00 |
| [LEDA 59741](../data/galaxy_sources/leda_59741.json) | barred spiral | 114.8450 | 369.801 | 0.02598 | 0.414 | 0.928 | 77.00 |
| [Z 212-14](../data/galaxy_sources/z_212_14.json) | spiral | 114.8889 | 369.941 | 0.02599 | 1.796 | 0.216 | 75.00 |
| [LEDA 86495](../data/galaxy_sources/leda_86495.json) | spiral | 114.9328 | 370.080 | 0.026 | 0.629 | 0.740 | 118.00 |
| [Z 72-32](../data/galaxy_sources/z_72_32.json) | spiral | 114.9328 | 370.080 | 0.026 | 0.510 | 0.428 | 51.00 |
| [NGC 3164](../data/galaxy_sources/ngc_3164.json) | spiral | 114.9768 | 370.220 | 0.02601 | 0.986 | 0.650 | 12.00 |
| [Z 405-9](../data/galaxy_sources/z_405_9.json) | spiral | 115.0185 | 370.353 | 0.0260195 | 0.622 | 0.741 | 146.00 |
| [Z 108-156](../data/galaxy_sources/z_108_156.json) | spiral | 115.0207 | 370.360 | 0.02602 | 0.636 | 0.639 | 50.00 |
| [LEDA 5795](../data/galaxy_sources/leda_5795.json) | spiral | 115.0266 | 370.378 | 0.02602134 | 0.462 | 0.438 | 76.00 |
| [Z 438-8](../data/galaxy_sources/z_438_8.json) | elliptical | 115.0646 | 370.499 | 0.02603 | 0.606 | 0.891 | 60.00 |
| [Z 451-1](../data/galaxy_sources/z_451_1.json) | spiral | 115.0787 | 370.544 | 0.02603319 | 0.833 | 0.419 | 165.00 |
| [LEDA 1094716](../data/galaxy_sources/leda_1094716.json) | barred spiral | 115.1086 | 370.639 | 0.02604 | 0.330 | 0.902 | 59.00 |
| [Z 50-32](../data/galaxy_sources/z_50_32.json) | barred spiral | 115.1086 | 370.639 | 0.02604 | 0.310 | 0.800 | 165.00 |
| [ESO 382-46](../data/galaxy_sources/eso_382_46.json) | spiral | 115.1087 | 370.640 | 0.02604004 | 0.600 | 0.280 | 20.00 |
| [2MASX J11072439-7025377](../data/galaxy_sources/galaxy_2masx_j11072439_7025377.json) | spiral | 115.1900 | 370.898 | 0.02605854 | 0.550 | 0.900 | 100.00 |
| [Z 18-102](../data/galaxy_sources/z_18_102.json) | lenticular | 115.1964 | 370.918 | 0.02606 | 0.613 | 0.624 | 47.00 |
| [LEDA 86990](../data/galaxy_sources/leda_86990.json) | spiral | 115.2403 | 371.058 | 0.02607 | 0.478 | 0.692 | 116.00 |
| [Z 7-11](../data/galaxy_sources/z_7_11.json) | spiral | 115.2843 | 371.197 | 0.02608 | 0.453 | 0.788 | 65.00 |
| [UGC 2266](../data/galaxy_sources/ugc_2266.json) | spiral | 115.3495 | 371.405 | 0.02609484 | 0.605 | 0.741 | 23.00 |
| [Z 287-53](../data/galaxy_sources/z_287_53.json) | spiral | 115.3721 | 371.477 | 0.0261 | 0.838 | 0.485 | 89.00 |
| [LEDA 2036716](../data/galaxy_sources/leda_2036716.json) | elliptical | 115.3721 | 371.477 | 0.0261 | 0.484 | 0.596 | 157.00 |
| [Z 268-27](../data/galaxy_sources/z_268_27.json) | lenticular | 115.4161 | 371.616 | 0.02611 | 0.707 | 0.454 | 175.00 |
| [Z 50-74](../data/galaxy_sources/z_50_74.json) | spiral | 115.4161 | 371.616 | 0.02611 | 0.872 | 0.499 | 34.00 |
| [Z 287-79](../data/galaxy_sources/z_287_79.json) | spiral | 115.4600 | 371.756 | 0.02612 | 0.577 | 0.520 | 55.00 |
| [LEDA 11983](../data/galaxy_sources/leda_11983.json) | spiral | 115.4600 | 371.756 | 0.02612 | 0.498 | 0.737 | 154.00 |
| [Z 50-80](../data/galaxy_sources/z_50_80.json) | spiral | 115.4600 | 371.756 | 0.02612 | 0.368 | 0.819 | 115.00 |
| [MCG+07-19-028](../data/galaxy_sources/mcg_07_19_028.json) | spiral | 115.5039 | 371.896 | 0.02613 | 0.583 | 0.428 | 37.00 |
| [Z 220-40](../data/galaxy_sources/z_220_40.json) | barred spiral | 115.5039 | 371.896 | 0.02613 | 0.891 | 0.643 | 96.00 |
| [Z 19-10](../data/galaxy_sources/z_19_10.json) | elliptical | 115.5039 | 371.896 | 0.02613 | 0.737 | 0.825 | 102.00 |
| [Z 438-9](../data/galaxy_sources/z_438_9.json) | lenticular | 115.5478 | 372.035 | 0.02614 | 0.761 | 0.858 | 123.00 |
| [LEDA 48968](../data/galaxy_sources/leda_48968.json) | spiral | 115.5478 | 372.035 | 0.02614 | 0.322 | 0.871 | 115.00 |
| [MCG+10-15-059](../data/galaxy_sources/mcg_10_15_059.json) | elliptical | 115.5918 | 372.175 | 0.02615 | 0.715 | 0.628 | 6.00 |
| [UGC 12355](../data/galaxy_sources/ugc_12355.json) | spiral | 115.6203 | 372.265 | 0.02615649 | 0.918 | 0.288 | 136.00 |
| [Z 299-56](../data/galaxy_sources/z_299_56.json) | spiral | 115.6357 | 372.314 | 0.02616 | 0.912 | 0.425 | 138.00 |
| [Z 102-62](../data/galaxy_sources/z_102_62.json) | lenticular | 115.6357 | 372.314 | 0.02616 | 0.685 | 0.948 | 160.00 |
| [LEDA 47245](../data/galaxy_sources/leda_47245.json) | spiral | 115.6504 | 372.361 | 0.02616334 | 0.854 | 0.285 | 136.00 |
| [Z 48-75](../data/galaxy_sources/z_48_75.json) | spiral | 115.6796 | 372.454 | 0.02617 | 0.862 | 0.203 | 70.00 |
| [LEDA 184633](../data/galaxy_sources/leda_184633.json) | spiral | 115.6796 | 372.454 | 0.02617 | 0.625 | 0.306 | 90.00 |
| [Z 247-18](../data/galaxy_sources/z_247_18.json) | spiral | 115.7235 | 372.594 | 0.02618 | 0.859 | 0.749 | 91.00 |
| [IC 1260](../data/galaxy_sources/ic_1260.json) | lenticular | 115.7675 | 372.733 | 0.02619 | 0.566 | 0.842 | 150.00 |
| [Z 456-32](../data/galaxy_sources/z_456_32.json) | spiral | 115.8008 | 372.839 | 0.0261976 | 0.829 | 0.413 | 135.00 |
| [Z 67-13](../data/galaxy_sources/z_67_13.json) | spiral | 115.8114 | 372.873 | 0.0262 | 0.583 | 0.796 | 132.00 |
| [NGC 7405](../data/galaxy_sources/ngc_7405.json) | spiral | 115.8159 | 372.887 | 0.02620102 | 0.443 | 0.667 | 97.00 |
| [Z 502-120](../data/galaxy_sources/z_502_120.json) | spiral | 115.8309 | 372.935 | 0.02620445 | 0.802 | 0.723 | 121.00 |
| [LEDA 22975](../data/galaxy_sources/leda_22975.json) | spiral | 115.8553 | 373.012 | 0.02621 | 0.565 | 0.248 | 50.00 |
| [Z 450-5](../data/galaxy_sources/z_450_5.json) | spiral | 115.8760 | 373.078 | 0.02621472 | 2.451 | 0.218 | 78.00 |
| [Z 36-51](../data/galaxy_sources/z_36_51.json) | spiral | 115.8992 | 373.152 | 0.02622 | 0.736 | 0.306 | 145.00 |
| [MCG+04-02-028](../data/galaxy_sources/mcg_04_02_028.json) | spiral | 115.9061 | 373.174 | 0.02622157 | 0.213 | 0.981 | 145.00 |
| [LEDA 5332](../data/galaxy_sources/leda_5332.json) | spiral | 115.9212 | 373.222 | 0.026225 | 0.470 | 0.568 | 17.00 |
| [UGC 9898](../data/galaxy_sources/ugc_9898.json) | spiral | 115.9432 | 373.292 | 0.02623 | 1.138 | 0.191 | 40.00 |
| [Z 463-3](../data/galaxy_sources/z_463_3.json) | spiral | 115.9814 | 373.413 | 0.0262387 | 0.782 | 0.413 | 75.00 |
| [IC 3850](../data/galaxy_sources/ic_3850.json) | elliptical | 115.9871 | 373.431 | 0.02624 | 0.840 | 0.673 | 41.00 |
| [LEDA 23060](../data/galaxy_sources/leda_23060.json) | spiral | 116.0310 | 373.571 | 0.02625 | 0.730 | 0.326 | 129.00 |
| [MCG+08-23-058](../data/galaxy_sources/mcg_08_23_058.json) | elliptical | 116.0310 | 373.571 | 0.02625 | 0.583 | 0.853 | 139.00 |
| [MCG+05-22-039](../data/galaxy_sources/mcg_05_22_039.json) | spiral | 116.0310 | 373.571 | 0.02625 | 0.373 | 0.780 | 125.00 |
| [2MASX J05105313+1802007](../data/galaxy_sources/galaxy_2masx_j05105313_1802007.json) | elliptical | 116.0416 | 373.604 | 0.0262524 | 0.290 | 0.921 | 75.00 |
| [Z 404-4](../data/galaxy_sources/z_404_4.json) | spiral | 116.0566 | 373.652 | 0.02625583 | 0.439 | 0.958 | 110.00 |
| [Z 436-25](../data/galaxy_sources/z_436_25.json) | spiral | 116.0716 | 373.700 | 0.02625925 | 0.588 | 0.617 | 5.00 |
| [Z 210-12](../data/galaxy_sources/z_210_12.json) | spiral | 116.0749 | 373.710 | 0.02626 | 0.669 | 0.401 | 147.00 |
| [Z 121-99](../data/galaxy_sources/z_121_99.json) | spiral | 116.0749 | 373.710 | 0.02626 | 0.588 | 0.889 | 86.00 |
| [Z 190-62](../data/galaxy_sources/z_190_62.json) | elliptical | 116.0749 | 373.710 | 0.02626 | 0.835 | 0.781 | 167.00 |
| [Z 38-23](../data/galaxy_sources/z_38_23.json) | weakly barred spiral | 116.1188 | 373.850 | 0.02627 | 0.934 | 0.495 | 69.00 |
| [Z 475-31](../data/galaxy_sources/z_475_31.json) | spiral | 116.1469 | 373.939 | 0.02627638 | 0.698 | 0.679 | 55.00 |
| [MCG+08-28-008](../data/galaxy_sources/mcg_08_28_008.json) | spiral | 116.1628 | 373.990 | 0.02628 | 0.699 | 0.565 | 178.00 |
| [MCG+10-20-060](../data/galaxy_sources/mcg_10_20_060.json) | lenticular | 116.2067 | 374.129 | 0.02629 | 0.499 | 0.846 | 160.00 |
| [Z 64-75](../data/galaxy_sources/z_64_75.json) | spiral | 116.2506 | 374.269 | 0.0263 | 1.270 | 0.324 | 64.00 |
| [Z 184-38](../data/galaxy_sources/z_184_38.json) | spiral | 116.2506 | 374.269 | 0.0263 | 1.008 | 0.499 | 27.00 |
| [LEDA 91237](../data/galaxy_sources/leda_91237.json) | spiral | 116.2506 | 374.269 | 0.0263 | 0.856 | 0.180 | 11.00 |
| [IC 4380](../data/galaxy_sources/ic_4380.json) | spiral | 116.2506 | 374.269 | 0.0263 | 0.678 | 0.904 | 145.00 |
| [LEDA 213831](../data/galaxy_sources/leda_213831.json) | spiral | 116.2506 | 374.269 | 0.0263 | 0.721 | 0.293 | 122.00 |
| [MCG+08-27-033](../data/galaxy_sources/mcg_08_27_033.json) | spiral | 116.2945 | 374.408 | 0.02631 | 0.750 | 0.723 | 96.00 |
| [Z 239-52](../data/galaxy_sources/z_239_52.json) | spiral | 116.3385 | 374.548 | 0.02632 | 0.829 | 0.715 | 108.00 |
| [MCG+05-22-006](../data/galaxy_sources/mcg_05_22_006.json) | spiral | 116.3385 | 374.548 | 0.02632 | 0.595 | 0.610 | 159.00 |
| [LEDA 84133](../data/galaxy_sources/leda_84133.json) | spiral | 116.3385 | 374.548 | 0.02632 | 0.581 | 0.306 | 137.00 |
| [Z 438-23](../data/galaxy_sources/z_438_23.json) | spiral | 116.3575 | 374.609 | 0.02632434 | 1.405 | 0.283 | 138.00 |
| [LEDA 85303](../data/galaxy_sources/leda_85303.json) | spiral | 116.3876 | 374.704 | 0.02633119 | 0.420 | 0.823 | 71.00 |
| [Z 136-52](../data/galaxy_sources/z_136_52.json) | spiral | 116.4263 | 374.827 | 0.02634 | 0.972 | 0.419 | 161.00 |
| [Z 102-65](../data/galaxy_sources/z_102_65.json) | spiral | 116.4629 | 374.943 | 0.02634832 | 1.293 | 0.335 | 84.00 |
| [Z 180-38](../data/galaxy_sources/z_180_38.json) | spiral | 116.4702 | 374.967 | 0.02635 | 0.755 | 0.667 | 133.00 |
| [Z 475-28](../data/galaxy_sources/z_475_28.json) | spiral | 116.4779 | 374.991 | 0.02635175 | 0.630 | 0.753 | 72.00 |
| [Z 288-21](../data/galaxy_sources/z_288_21.json) | spiral | 116.5141 | 375.106 | 0.02636 | 0.558 | 0.690 | 87.00 |
| [Z 207-31](../data/galaxy_sources/z_207_31.json) | spiral | 116.5141 | 375.106 | 0.02636 | 1.441 | 0.619 | 51.00 |
| [LEDA 86440](../data/galaxy_sources/leda_86440.json) | spiral | 116.5141 | 375.106 | 0.02636 | 0.858 | 0.236 | 125.00 |
| [Z 34-3](../data/galaxy_sources/z_34_3.json) | spiral | 116.5682 | 375.278 | 0.0263723 | 0.554 | 0.282 | 40.00 |
| [LEDA 2043](../data/galaxy_sources/leda_2043.json) | elliptical | 116.5983 | 375.374 | 0.02637915 | 0.503 | 0.557 | 15.00 |
| [MCG+08-18-003](../data/galaxy_sources/mcg_08_18_003.json) | spiral | 116.6459 | 375.525 | 0.02639 | 0.564 | 0.501 | 129.00 |
| [Z 452-6](../data/galaxy_sources/z_452_6.json) | spiral | 116.7187 | 375.756 | 0.02640656 | 0.703 | 0.666 | 69.00 |
| [Z 428-13](../data/galaxy_sources/z_428_13.json) | spiral | 116.7337 | 375.804 | 0.02640999 | 0.843 | 0.636 | 143.00 |
| [Z 238-50](../data/galaxy_sources/z_238_50.json) | spiral | 116.7338 | 375.804 | 0.02641 | 0.761 | 0.392 | 137.00 |
| [Z 268-84](../data/galaxy_sources/z_268_84.json) | spiral | 116.7338 | 375.804 | 0.02641 | 0.936 | 0.373 | 7.00 |
| [LEDA 23000](../data/galaxy_sources/leda_23000.json) | barred spiral | 116.7777 | 375.944 | 0.02642 | 0.560 | 0.513 | 83.00 |
| [MCG+00-09-012](../data/galaxy_sources/mcg_00_09_012.json) | spiral | 116.7777 | 375.944 | 0.02642 | 0.570 | 0.559 | 180.00 |
| [LEDA 22984](../data/galaxy_sources/leda_22984.json) | spiral | 116.7777 | 375.944 | 0.02642 | 0.230 | 0.961 | 55.00 |
| [LEDA 87106](../data/galaxy_sources/leda_87106.json) | spiral | 116.7777 | 375.944 | 0.02642 | 0.467 | 0.843 | 19.00 |
| [Z 39-54](../data/galaxy_sources/z_39_54.json) | spiral | 116.7777 | 375.944 | 0.02642 | 0.620 | 0.494 | 114.00 |
| [ESO 13-14](../data/galaxy_sources/eso_13_14.json) | spiral | 116.7803 | 375.952 | 0.02642061 | 0.427 | 0.759 | 135.00 |
| [LEDA 84062](../data/galaxy_sources/leda_84062.json) | spiral | 116.8216 | 376.083 | 0.02643 | 0.682 | 0.520 | 170.00 |
| [MCG+02-07-002](../data/galaxy_sources/mcg_02_07_002.json) | spiral | 116.8992 | 376.330 | 0.02644768 | 0.846 | 0.632 | 140.00 |
| [IC 3615](../data/galaxy_sources/ic_3615.json) | spiral | 116.9094 | 376.362 | 0.02645 | 1.129 | 0.304 | 12.00 |
| [LEDA 82558](../data/galaxy_sources/leda_82558.json) | spiral | 116.9534 | 376.502 | 0.02646 | 0.743 | 0.326 | 132.00 |
| [Z 148-50](../data/galaxy_sources/z_148_50.json) | spiral | 116.9534 | 376.502 | 0.02646 | 0.585 | 0.638 | 49.00 |
| [Z 74-116](../data/galaxy_sources/z_74_116.json) | spiral | 116.9534 | 376.502 | 0.02646 | 0.941 | 0.250 | 29.00 |
| [LEDA 51320](../data/galaxy_sources/leda_51320.json) | lenticular | 116.9973 | 376.641 | 0.02647 | 0.583 | 0.790 | 4.00 |
| [Z 20-23](../data/galaxy_sources/z_20_23.json) | spiral | 116.9973 | 376.641 | 0.02647 | 0.628 | 0.513 | 81.00 |
| [MCG+08-18-009](../data/galaxy_sources/mcg_08_18_009.json) | barred spiral | 117.0851 | 376.921 | 0.02649 | 0.610 | 0.717 | 74.00 |
| [Z 432-36](../data/galaxy_sources/z_432_36.json) | spiral | 117.0851 | 376.921 | 0.02649 | 0.619 | 0.773 | 111.00 |
| [Z 93-22](../data/galaxy_sources/z_93_22.json) | spiral | 117.0948 | 376.951 | 0.02649222 | 0.900 | 0.668 | 100.00 |
| [Z 124-36](../data/galaxy_sources/z_124_36.json) | spiral | 117.0948 | 376.951 | 0.02649222 | 0.639 | 0.809 | 29.00 |
| [Z 438-22](../data/galaxy_sources/z_438_22.json) | spiral | 117.1400 | 377.095 | 0.0265025 | 1.160 | 0.564 | 45.00 |
| [Z 248-52](../data/galaxy_sources/z_248_52.json) | spiral | 117.1729 | 377.200 | 0.02651 | 0.875 | 0.778 | 159.00 |
| [Z 495-10](../data/galaxy_sources/z_495_10.json) | spiral | 117.2002 | 377.286 | 0.0265162 | 1.203 | 0.361 | 63.00 |
| [Z 268-56](../data/galaxy_sources/z_268_56.json) | elliptical | 117.2169 | 377.339 | 0.02652 | 0.581 | 0.735 | 95.00 |
| [LEDA 126756](../data/galaxy_sources/leda_126756.json) | elliptical | 117.2608 | 377.479 | 0.02653 | 0.400 | 0.391 | 159.00 |
| [Z 503-17](../data/galaxy_sources/z_503_17.json) | spiral | 117.2754 | 377.525 | 0.02653333 | 1.065 | 0.615 | 102.00 |
| [Z 38-44](../data/galaxy_sources/z_38_44.json) | spiral | 117.3047 | 377.618 | 0.02654 | 0.618 | 0.810 | 40.00 |
| [Z 101-14](../data/galaxy_sources/z_101_14.json) | spiral | 117.3047 | 377.618 | 0.02654 | 0.746 | 0.357 | 67.00 |
| [Z 499-62](../data/galaxy_sources/z_499_62.json) | spiral | 117.3356 | 377.716 | 0.02654704 | 0.867 | 0.200 | 5.00 |
| [Z 63-93](../data/galaxy_sources/z_63_93.json) | spiral | 117.3486 | 377.758 | 0.02655 | 0.373 | 0.721 | 80.00 |
| [LEDA 82413](../data/galaxy_sources/leda_82413.json) | spiral | 117.4365 | 378.037 | 0.02657 | 0.503 | 0.499 | 117.00 |
| [NGC 7641](../data/galaxy_sources/ngc_7641.json) | spiral | 117.4410 | 378.051 | 0.02657103 | 1.750 | 0.370 | 133.00 |
| [MCG+03-05-007](../data/galaxy_sources/mcg_03_05_007.json) | spiral | 117.4861 | 378.195 | 0.02658131 | 0.635 | 0.976 | 169.00 |
| [Z 267-40](../data/galaxy_sources/z_267_40.json) | spiral | 117.5243 | 378.316 | 0.02659 | 0.528 | 0.565 | 80.00 |
| [Z 288-24](../data/galaxy_sources/z_288_24.json) | barred spiral | 117.5682 | 378.455 | 0.0266 | 0.470 | 0.595 | 92.00 |
| [Z 36-26](../data/galaxy_sources/z_36_26.json) | spiral | 117.5682 | 378.455 | 0.0266 | 0.859 | 0.368 | 66.00 |
| [LEDA 2029644](../data/galaxy_sources/leda_2029644.json) | elliptical | 117.6121 | 378.595 | 0.02661 | 0.365 | 0.536 | 175.00 |
| [Z 431-45](../data/galaxy_sources/z_431_45.json) | spiral | 117.6366 | 378.673 | 0.02661557 | 1.058 | 0.289 | 169.00 |
| [IC 479](../data/galaxy_sources/ic_479.json) | spiral | 117.6560 | 378.734 | 0.02662 | 0.618 | 0.837 | 166.00 |
| [NGC 7434](../data/galaxy_sources/ngc_7434.json) | elliptical | 117.6560 | 378.734 | 0.02662 | 0.891 | 0.541 | 76.00 |
| [UGC 1683](../data/galaxy_sources/ugc_1683.json) | spiral | 117.6968 | 378.864 | 0.02662928 | 1.059 | 0.170 | 2.00 |
| [Z 247-27](../data/galaxy_sources/z_247_27.json) | spiral | 117.6999 | 378.874 | 0.02663 | 0.714 | 0.784 | 47.00 |
| [UGC 9651](../data/galaxy_sources/ugc_9651.json) | barred spiral | 117.7439 | 379.014 | 0.02664 | 1.273 | 0.279 | 115.00 |
| [Z 524-48](../data/galaxy_sources/z_524_48.json) | spiral | 117.7871 | 379.151 | 0.02664984 | 0.950 | 0.380 | 30.00 |
| [IC 2400](../data/galaxy_sources/ic_2400.json) | spiral | 117.7878 | 379.153 | 0.02665 | 0.744 | 0.465 | 113.00 |
| [Z 503-10](../data/galaxy_sources/z_503_10.json) | spiral | 117.8172 | 379.246 | 0.0266567 | 0.549 | 0.758 | 157.00 |
| [Z 477-32](../data/galaxy_sources/z_477_32.json) | spiral | 117.8774 | 379.438 | 0.0266704 | 0.918 | 0.264 | 75.00 |
| [Z 102-29](../data/galaxy_sources/z_102_29.json) | barred spiral | 117.9195 | 379.572 | 0.02668 | 0.899 | 0.335 | 3.00 |
| [Z 75-26](../data/galaxy_sources/z_75_26.json) | spiral | 117.9225 | 379.581 | 0.02668068 | 0.649 | 0.379 | 129.00 |
| [Z 455-67](../data/galaxy_sources/z_455_67.json) | barred spiral | 117.9376 | 379.629 | 0.02668411 | 1.088 | 0.389 | 136.00 |
| [Z 209-10](../data/galaxy_sources/z_209_10.json) | spiral | 117.9634 | 379.711 | 0.02669 | 0.622 | 0.571 | 73.00 |
| [IC 482](../data/galaxy_sources/ic_482.json) | spiral | 117.9634 | 379.711 | 0.02669 | 0.667 | 0.814 | 140.00 |
| [Z 456-31](../data/galaxy_sources/z_456_31.json) | barred spiral | 117.9827 | 379.772 | 0.02669439 | 0.548 | 0.869 | 97.00 |
| [MCG+08-18-008](../data/galaxy_sources/mcg_08_18_008.json) | spiral | 118.0074 | 379.851 | 0.0267 | 0.825 | 0.273 | 140.00 |
| [IC 3484](../data/galaxy_sources/ic_3484.json) | spiral | 118.0074 | 379.851 | 0.0267 | 0.840 | 0.702 | 35.00 |
| [MCG+07-28-011](../data/galaxy_sources/mcg_07_28_011.json) | spiral | 118.0074 | 379.851 | 0.0267 | 0.541 | 0.681 | 21.00 |
| [LEDA 1154100](../data/galaxy_sources/leda_1154100.json) | lenticular | 118.0074 | 379.851 | 0.0267 | 0.416 | 0.914 | 131.00 |
| [Z 287-76](../data/galaxy_sources/z_287_76.json) | elliptical | 118.0952 | 380.130 | 0.02672 | 0.825 | 0.920 | 93.00 |
| [LEDA 82329](../data/galaxy_sources/leda_82329.json) | spiral | 118.1391 | 380.269 | 0.02673 | 0.560 | 0.541 | 100.00 |
| [Z 438-18](../data/galaxy_sources/z_438_18.json) | spiral | 118.1483 | 380.298 | 0.02673209 | 0.945 | 0.234 | 52.00 |
| [ESO 12-12](../data/galaxy_sources/eso_12_12.json) | spiral | 118.1724 | 380.375 | 0.02673757 | 0.713 | 0.961 | 40.00 |
| [Z 376-27](../data/galaxy_sources/z_376_27.json) | elliptical | 118.1830 | 380.409 | 0.02674 | 0.627 | 0.810 | 72.00 |
| [Z 248-40](../data/galaxy_sources/z_248_40.json) | spiral | 118.2269 | 380.548 | 0.02675 | 0.757 | 0.443 | 157.00 |
| [Z 437-46](../data/galaxy_sources/z_437_46.json) | spiral | 118.2536 | 380.633 | 0.02675608 | 0.543 | 0.804 | 48.00 |
| [Z 150-55](../data/galaxy_sources/z_150_55.json) | spiral | 118.2708 | 380.688 | 0.02676 | 0.862 | 0.460 | 64.00 |
| [Z 34-1](../data/galaxy_sources/z_34_1.json) | spiral | 118.2708 | 380.688 | 0.02676 | 0.628 | 0.911 | 66.00 |
| [LEDA 35609](../data/galaxy_sources/leda_35609.json) | lenticular | 118.3147 | 380.827 | 0.02677 | 0.460 | 0.650 | 172.00 |
| [LEDA 32407](../data/galaxy_sources/leda_32407.json) | spiral | 118.3147 | 380.827 | 0.02677 | 0.303 | 0.739 | 13.00 |
| [LEDA 83557](../data/galaxy_sources/leda_83557.json) | spiral | 118.3147 | 380.827 | 0.02677 | 0.578 | 0.309 | 34.00 |
| [SDSS J024050.03-075832.5](../data/galaxy_sources/sdss_j024050_03_075832_5.json) | barred spiral | 118.3587 | 380.967 | 0.02678 | 0.337 | 0.899 | 115.00 |
| [UGC 12449](../data/galaxy_sources/ugc_12449.json) | spiral | 118.3785 | 381.030 | 0.02678453 | 1.079 | 0.185 | 142.00 |
| [Z 264-100](../data/galaxy_sources/z_264_100.json) | spiral | 118.4026 | 381.106 | 0.02679 | 0.437 | 0.620 | 50.00 |
| [Z 150-9](../data/galaxy_sources/z_150_9.json) | spiral | 118.4026 | 381.106 | 0.02679 | 0.570 | 0.306 | 102.00 |
| [IC 3079](../data/galaxy_sources/ic_3079.json) | spiral | 118.4026 | 381.106 | 0.02679 | 0.712 | 0.670 | 16.00 |
| [Z 433-17](../data/galaxy_sources/z_433_17.json) | spiral | 118.4041 | 381.111 | 0.02679036 | 0.529 | 0.444 | 82.00 |
| [Z 287-67](../data/galaxy_sources/z_287_67.json) | barred spiral | 118.4465 | 381.246 | 0.0268 | 0.602 | 0.662 | 99.00 |
| [Z 148-48](../data/galaxy_sources/z_148_48.json) | spiral | 118.4904 | 381.385 | 0.02681 | 0.655 | 0.569 | 103.00 |
| [Z 95-88](../data/galaxy_sources/z_95_88.json) | spiral | 118.4904 | 381.385 | 0.02681 | 0.347 | 0.741 | 160.00 |
| [MCG+09-19-105](../data/galaxy_sources/mcg_09_19_105.json) | spiral | 118.4904 | 381.385 | 0.02681 | 0.618 | 0.466 | 24.00 |
| [Z 268-14](../data/galaxy_sources/z_268_14.json) | elliptical | 118.5343 | 381.524 | 0.02682 | 0.732 | 0.797 | 76.00 |
| [IC 152](../data/galaxy_sources/ic_152.json) | spiral | 118.5343 | 381.524 | 0.02682 | 0.478 | 0.740 | 118.00 |
| [Z 102-43](../data/galaxy_sources/z_102_43.json) | spiral | 118.5396 | 381.541 | 0.0268212 | 0.457 | 0.755 | 0.00 |
| [Z 181-53](../data/galaxy_sources/z_181_53.json) | lenticular | 118.5782 | 381.664 | 0.02683 | 0.517 | 0.480 | 130.00 |
| [Z 428-43](../data/galaxy_sources/z_428_43.json) | barred spiral | 118.5847 | 381.685 | 0.02683148 | 1.052 | 0.621 | 53.00 |
| [Z 245-39](../data/galaxy_sources/z_245_39.json) | elliptical | 118.6221 | 381.803 | 0.02684 | 1.025 | 0.635 | 133.00 |
| [LEDA 1017079](../data/galaxy_sources/leda_1017079.json) | lenticular | 118.6221 | 381.803 | 0.02684 | 0.415 | 0.878 | 1.00 |
| [Z 428-9](../data/galaxy_sources/z_428_9.json) | spiral | 118.6600 | 381.924 | 0.02684862 | 0.669 | 0.871 | 162.00 |
| [Z 273-34](../data/galaxy_sources/z_273_34.json) | spiral | 118.6660 | 381.943 | 0.02685 | 0.737 | 0.579 | 139.00 |
| [Z 239-7](../data/galaxy_sources/z_239_7.json) | weakly barred spiral | 118.6660 | 381.943 | 0.02685 | 0.633 | 0.933 | 86.00 |
| [ESO 382-18](../data/galaxy_sources/eso_382_18.json) | spiral | 118.7051 | 382.067 | 0.02685891 | 0.333 | 0.781 | 140.00 |
| [UGC 8661](../data/galaxy_sources/ugc_8661.json) | lenticular | 118.7099 | 382.082 | 0.02686 | 0.836 | 0.665 | 95.00 |
| [Z 48-5](../data/galaxy_sources/z_48_5.json) | spiral | 118.7099 | 382.082 | 0.02686 | 0.783 | 0.274 | 103.00 |
| [MCG+08-24-099](../data/galaxy_sources/mcg_08_24_099.json) | spiral | 118.7539 | 382.222 | 0.02687 | 0.544 | 0.851 | 31.00 |
| [Z 151-9](../data/galaxy_sources/z_151_9.json) | spiral | 118.7539 | 382.222 | 0.02687 | 0.733 | 0.786 | 98.00 |
| [NGC 3337](../data/galaxy_sources/ngc_3337.json) | lenticular | 118.7539 | 382.222 | 0.02687 | 0.994 | 0.504 | 51.00 |
| [LEDA 1162940](../data/galaxy_sources/leda_1162940.json) | spiral | 118.7539 | 382.222 | 0.02687 | 0.676 | 0.219 | 49.00 |
| [Z 247-38](../data/galaxy_sources/z_247_38.json) | spiral | 118.7978 | 382.361 | 0.02688 | 0.413 | 0.479 | 75.00 |
| [Z 33-31](../data/galaxy_sources/z_33_31.json) | lenticular | 118.7978 | 382.361 | 0.02688 | 0.373 | 0.869 | 89.00 |
| [MCG+06-25-027](../data/galaxy_sources/mcg_06_25_027.json) | lenticular | 118.7978 | 382.361 | 0.02688 | 0.627 | 0.649 | 91.00 |
| [IC 4355](../data/galaxy_sources/ic_4355.json) | spiral | 118.7978 | 382.361 | 0.02688 | 0.435 | 0.900 | 164.00 |
| [MCG-01-53-004](../data/galaxy_sources/mcg_01_53_004.json) | spiral | 118.7978 | 382.361 | 0.02688 | 0.905 | 0.760 | 110.00 |
| [Z 505-52](../data/galaxy_sources/z_505_52.json) | spiral | 118.8105 | 382.402 | 0.0268829 | 0.563 | 0.501 | 120.00 |
| [LEDA 3087347](../data/galaxy_sources/leda_3087347.json) | spiral | 118.8417 | 382.501 | 0.02689 | 0.518 | 0.878 | 12.00 |
| [LEDA 82405](../data/galaxy_sources/leda_82405.json) | spiral | 118.8856 | 382.640 | 0.0269 | 0.438 | 0.526 | 39.00 |
| [Z 152-28](../data/galaxy_sources/z_152_28.json) | lenticular | 118.8856 | 382.640 | 0.0269 | 0.734 | 0.892 | 72.00 |
| [Z 66-36](../data/galaxy_sources/z_66_36.json) | spiral | 118.8856 | 382.640 | 0.0269 | 0.776 | 0.808 | 111.00 |
| [Z 40-32](../data/galaxy_sources/z_40_32.json) | weakly barred spiral | 118.9295 | 382.780 | 0.02691 | 0.537 | 0.974 | 117.00 |
| [LEDA 257](../data/galaxy_sources/leda_257.json) | spiral | 118.9309 | 382.784 | 0.02691032 | 0.681 | 0.312 | 142.00 |
| [Z 151-58](../data/galaxy_sources/z_151_58.json) | lenticular | 118.9734 | 382.919 | 0.02692 | 0.895 | 0.644 | 16.00 |
| [Z 94-74](../data/galaxy_sources/z_94_74.json) | barred spiral | 118.9734 | 382.919 | 0.02692 | 0.840 | 0.723 | 15.00 |
| [Z 375-16](../data/galaxy_sources/z_375_16.json) | lenticular | 119.0173 | 383.059 | 0.02693 | 0.461 | 0.861 | 29.00 |
| [MCG+11-16-011](../data/galaxy_sources/mcg_11_16_011.json) | barred spiral | 119.0612 | 383.198 | 0.02694 | 0.573 | 0.814 | 10.00 |
| [Z 75-47](../data/galaxy_sources/z_75_47.json) | spiral | 119.0612 | 383.198 | 0.02694 | 0.638 | 0.758 | 176.00 |
| [LEDA 3092531](../data/galaxy_sources/leda_3092531.json) | lenticular | 119.0612 | 383.198 | 0.02694 | 0.470 | 0.799 | 72.00 |
| [Z 433-21](../data/galaxy_sources/z_433_21.json) | barred spiral | 119.0814 | 383.262 | 0.0269446 | 1.140 | 0.605 | 22.00 |
| [Z 66-96](../data/galaxy_sources/z_66_96.json) | spiral | 119.1051 | 383.338 | 0.02695 | 0.856 | 0.770 | 7.00 |
| [Z 218-22](../data/galaxy_sources/z_218_22.json) | spiral | 119.1051 | 383.338 | 0.02695 | 0.624 | 0.738 | 73.00 |
| [Z 74-3](../data/galaxy_sources/z_74_3.json) | spiral | 119.1051 | 383.338 | 0.02695 | 0.595 | 0.533 | 116.00 |
| [MCG+09-17-041](../data/galaxy_sources/mcg_09_17_041.json) | barred spiral | 119.1490 | 383.477 | 0.02696 | 0.495 | 0.507 | 145.00 |
| [MCG+07-19-029](../data/galaxy_sources/mcg_07_19_029.json) | barred spiral | 119.1490 | 383.477 | 0.02696 | 0.513 | 0.795 | 112.00 |
| [MCG+02-55-030](../data/galaxy_sources/mcg_02_55_030.json) | spiral | 119.1490 | 383.477 | 0.02696 | 0.751 | 0.332 | 44.00 |
| [Z 38-130](../data/galaxy_sources/z_38_130.json) | spiral | 119.1490 | 383.477 | 0.02696 | 0.890 | 0.394 | 122.00 |
| [Z 268-53](../data/galaxy_sources/z_268_53.json) | elliptical | 119.1930 | 383.616 | 0.02697 | 1.228 | 0.741 | 100.00 |
| [ESO 468-17](../data/galaxy_sources/eso_468_17.json) | spiral | 119.1930 | 383.616 | 0.02697 | 0.640 | 0.359 | 150.00 |
| [Z 156-38](../data/galaxy_sources/z_156_38.json) | spiral | 119.1930 | 383.616 | 0.02697 | 0.744 | 0.471 | 123.00 |
| [Z 403-24](../data/galaxy_sources/z_403_24.json) | spiral | 119.2169 | 383.692 | 0.02697545 | 0.819 | 0.657 | 70.00 |
| [MCG+08-24-100](../data/galaxy_sources/mcg_08_24_100.json) | lenticular | 119.2369 | 383.756 | 0.02698 | 0.486 | 0.880 | 57.00 |
| [Z 381-55](../data/galaxy_sources/z_381_55.json) | weakly barred spiral | 119.2369 | 383.756 | 0.02698 | 1.952 | 0.409 | 51.00 |
| [Z 151-75](../data/galaxy_sources/z_151_75.json) | spiral | 119.2369 | 383.756 | 0.02698 | 0.491 | 0.923 | 99.00 |
| [IC 2871](../data/galaxy_sources/ic_2871.json) | spiral | 119.2369 | 383.756 | 0.02698 | 0.800 | 0.743 | 81.00 |
| [MCG-01-53-007](../data/galaxy_sources/mcg_01_53_007.json) | spiral | 119.2369 | 383.756 | 0.02698 | 0.801 | 0.690 | 92.00 |
| [Z 220-48](../data/galaxy_sources/z_220_48.json) | lenticular | 119.2808 | 383.895 | 0.02699 | 0.609 | 0.727 | 168.00 |
| [Z 66-62](../data/galaxy_sources/z_66_62.json) | spiral | 119.2808 | 383.895 | 0.02699 | 0.587 | 0.710 | 142.00 |
| [MCG+09-16-007](../data/galaxy_sources/mcg_09_16_007.json) | spiral | 119.3247 | 384.035 | 0.027 | 0.851 | 0.213 | 143.00 |
| [LEDA 82403](../data/galaxy_sources/leda_82403.json) | spiral | 119.3247 | 384.035 | 0.027 | 0.783 | 0.371 | 141.00 |
| [NPM1G +39.0163](../data/galaxy_sources/npm1g_39_0163.json) | elliptical | 119.3686 | 384.174 | 0.02701 | 0.435 | 0.934 | 83.00 |
| [Z 209-22](../data/galaxy_sources/z_209_22.json) | barred spiral | 119.3686 | 384.174 | 0.02701 | 0.626 | 0.883 | 66.00 |
| [LEDA 89737](../data/galaxy_sources/leda_89737.json) | spiral | 119.3686 | 384.174 | 0.02701 | 0.481 | 0.625 | 30.00 |
| [LEDA 3092873](../data/galaxy_sources/leda_3092873.json) | lenticular | 119.4125 | 384.314 | 0.02702 | 0.596 | 0.724 | 79.00 |
| [ESO 364-12](../data/galaxy_sources/eso_364_12.json) | spiral | 119.4758 | 384.515 | 0.02703442 | 1.183 | 0.928 | 124.00 |
| [Z 153-7](../data/galaxy_sources/z_153_7.json) | spiral | 119.5003 | 384.593 | 0.02704 | 0.837 | 0.674 | 158.00 |
| [UGC 2214](../data/galaxy_sources/ugc_2214.json) | spiral | 119.5029 | 384.601 | 0.02704059 | 1.085 | 0.421 | 12.00 |
| [Z 475-53](../data/galaxy_sources/z_475_53.json) | spiral | 119.5781 | 384.840 | 0.02705773 | 0.901 | 0.347 | 134.00 |
| [NGC 6317](../data/galaxy_sources/ngc_6317.json) | spiral | 119.6320 | 385.011 | 0.02707 | 0.916 | 0.545 | 43.00 |
| [MCG+09-19-119](../data/galaxy_sources/mcg_09_19_119.json) | barred spiral | 119.6320 | 385.011 | 0.02707 | 0.870 | 0.626 | 160.00 |
| [Z 150-42](../data/galaxy_sources/z_150_42.json) | spiral | 119.6320 | 385.011 | 0.02707 | 0.878 | 0.284 | 173.00 |
| [NGC 3601](../data/galaxy_sources/ngc_3601.json) | spiral | 119.6320 | 385.011 | 0.02707 | 0.720 | 0.776 | 8.00 |
| [Z 68-6](../data/galaxy_sources/z_68_6.json) | spiral | 119.6320 | 385.011 | 0.02707 | 0.431 | 0.854 | 68.00 |
| [2MASX J05104713+1801490](../data/galaxy_sources/galaxy_2masx_j05104713_1801490.json) | elliptical | 119.6384 | 385.031 | 0.02707144 | 0.510 | 0.900 | 170.00 |
| [Z 152-40](../data/galaxy_sources/z_152_40.json) | spiral | 119.6759 | 385.150 | 0.02708 | 0.504 | 0.909 | 5.00 |
| [Z 150-53](../data/galaxy_sources/z_150_53.json) | weakly barred spiral | 119.7136 | 385.270 | 0.02708858 | 1.404 | 0.470 | 59.00 |
| [Z 221-6](../data/galaxy_sources/z_221_6.json) | spiral | 119.7198 | 385.290 | 0.02709 | 0.832 | 0.304 | 105.00 |
| [Z 428-73](../data/galaxy_sources/z_428_73.json) | spiral | 119.7738 | 385.461 | 0.0271023 | 0.769 | 0.620 | 160.00 |
| [LEDA 86863](../data/galaxy_sources/leda_86863.json) | spiral | 119.8076 | 385.569 | 0.02711 | 0.950 | 0.216 | 141.00 |
| [LEDA 72810](../data/galaxy_sources/leda_72810.json) | elliptical | 119.8076 | 385.569 | 0.02711 | 0.230 | 0.778 | 140.00 |
| [Z 48-54](../data/galaxy_sources/z_48_54.json) | spiral | 119.8076 | 385.569 | 0.02711 | 0.933 | 0.357 | 90.00 |
| [NGC 7601](../data/galaxy_sources/ngc_7601.json) | barred spiral | 119.8340 | 385.652 | 0.02711601 | 1.020 | 0.785 | 93.00 |
| [Z 249-18](../data/galaxy_sources/z_249_18.json) | lenticular | 119.8516 | 385.708 | 0.02712 | 0.461 | 0.893 | 62.00 |
| [Z 36-49](../data/galaxy_sources/z_36_49.json) | elliptical | 119.8516 | 385.708 | 0.02712 | 0.497 | 0.815 | 147.00 |
| [Z 476-48](../data/galaxy_sources/z_476_48.json) | spiral | 119.8943 | 385.844 | 0.02712973 | 0.863 | 0.388 | 3.00 |
| [Z 268-33](../data/galaxy_sources/z_268_33.json) | barred spiral | 119.8955 | 385.847 | 0.02713 | 0.531 | 0.559 | 11.00 |
| [LEDA 43741](../data/galaxy_sources/leda_43741.json) | spiral | 119.8955 | 385.847 | 0.02713 | 0.516 | 0.275 | 15.00 |
| [UGC 1606](../data/galaxy_sources/ugc_1606.json) | spiral | 119.9244 | 385.939 | 0.02713658 | 0.655 | 0.782 | 11.00 |
| [Z 516-3](../data/galaxy_sources/z_516_3.json) | spiral | 119.9244 | 385.939 | 0.02713658 | 0.527 | 0.419 | 105.00 |
| [Z 121-81](../data/galaxy_sources/z_121_81.json) | elliptical | 119.9394 | 385.987 | 0.02714 | 0.808 | 0.674 | 122.00 |
| [Z 152-31](../data/galaxy_sources/z_152_31.json) | spiral | 119.9394 | 385.987 | 0.02714 | 1.021 | 0.607 | 139.00 |
| [IC 4322](../data/galaxy_sources/ic_4322.json) | elliptical | 119.9394 | 385.987 | 0.02714 | 0.763 | 0.895 | 84.00 |
| [Z 148-9](../data/galaxy_sources/z_148_9.json) | spiral | 119.9833 | 386.126 | 0.02715 | 0.811 | 0.634 | 3.00 |
| [LEDA 28476](../data/galaxy_sources/leda_28476.json) | spiral | 120.0272 | 386.266 | 0.02716 | 0.488 | 0.334 | 143.00 |
| [Z 95-72](../data/galaxy_sources/z_95_72.json) | spiral | 120.0272 | 386.266 | 0.02716 | 0.804 | 0.437 | 64.00 |
| [Z 75-81](../data/galaxy_sources/z_75_81.json) | spiral | 120.0272 | 386.266 | 0.02716 | 0.757 | 0.567 | 41.00 |
| [MCG+10-19-030](../data/galaxy_sources/mcg_10_19_030.json) | spiral | 120.0711 | 386.405 | 0.02717 | 0.585 | 0.719 | 46.00 |
| [LEDA 25174](../data/galaxy_sources/leda_25174.json) | spiral | 120.0711 | 386.405 | 0.02717 | 0.565 | 0.402 | 4.00 |
| [MCG+06-22-022](../data/galaxy_sources/mcg_06_22_022.json) | barred spiral | 120.0711 | 386.405 | 0.02717 | 0.573 | 0.819 | 3.00 |
| [MCG+08-26-018](../data/galaxy_sources/mcg_08_26_018.json) | spiral | 120.1150 | 386.544 | 0.02718 | 0.574 | 0.785 | 89.00 |
| [Z 88-41](../data/galaxy_sources/z_88_41.json) | barred spiral | 120.1150 | 386.544 | 0.02718 | 0.683 | 0.485 | 162.00 |
| [MCG+09-19-061](../data/galaxy_sources/mcg_09_19_061.json) | spiral | 120.1589 | 386.684 | 0.02719 | 0.475 | 0.683 | 117.00 |
| [Z 238-5](../data/galaxy_sources/z_238_5.json) | spiral | 120.1589 | 386.684 | 0.02719 | 0.513 | 0.845 | 67.00 |
| [Z 185-20](../data/galaxy_sources/z_185_20.json) | spiral | 120.1589 | 386.684 | 0.02719 | 0.607 | 0.823 | 11.00 |
| [LEDA 85243](../data/galaxy_sources/leda_85243.json) | spiral | 120.1652 | 386.704 | 0.02719144 | 0.776 | 0.281 | 43.00 |
| [MCG+09-21-046](../data/galaxy_sources/mcg_09_21_046.json) | spiral | 120.2028 | 386.823 | 0.0272 | 0.998 | 0.281 | 17.00 |
| [Z 12-83](../data/galaxy_sources/z_12_83.json) | spiral | 120.2028 | 386.823 | 0.0272 | 0.545 | 0.690 | 115.00 |
| [Z 461-12](../data/galaxy_sources/z_461_12.json) | spiral | 120.2405 | 386.943 | 0.02720859 | 0.837 | 0.666 | 31.00 |
| [Z 287-43](../data/galaxy_sources/z_287_43.json) | barred spiral | 120.2467 | 386.963 | 0.02721 | 0.665 | 0.934 | 103.00 |
| [LEDA 82404](../data/galaxy_sources/leda_82404.json) | spiral | 120.2467 | 386.963 | 0.02721 | 0.502 | 0.643 | 8.00 |
| [Z 121-43](../data/galaxy_sources/z_121_43.json) | elliptical | 120.2467 | 386.963 | 0.02721 | 0.773 | 0.642 | 68.00 |
| [IC 474](../data/galaxy_sources/ic_474.json) | spiral | 120.2906 | 387.102 | 0.02722 | 1.156 | 0.328 | 112.00 |
| [Z 415-14](../data/galaxy_sources/z_415_14.json) | spiral | 120.3609 | 387.325 | 0.02723602 | 0.988 | 0.650 | 84.00 |
| [Z 181-50](../data/galaxy_sources/z_181_50.json) | barred spiral | 120.3784 | 387.381 | 0.02724 | 0.898 | 0.472 | 83.00 |
| [MCG+09-19-069](../data/galaxy_sources/mcg_09_19_069.json) | spiral | 120.4662 | 387.660 | 0.02726 | 1.056 | 0.217 | 36.00 |
| [Z 162-46](../data/galaxy_sources/z_162_46.json) | spiral | 120.4662 | 387.660 | 0.02726 | 0.829 | 0.707 | 78.00 |
| [LEDA 82437](../data/galaxy_sources/leda_82437.json) | spiral | 120.5101 | 387.799 | 0.02727 | 0.765 | 0.267 | 30.00 |
| [Z 7-39](../data/galaxy_sources/z_7_39.json) | spiral | 120.5101 | 387.799 | 0.02727 | 0.518 | 0.605 | 53.00 |
| [Z 40-18](../data/galaxy_sources/z_40_18.json) | spiral | 120.5540 | 387.938 | 0.02728 | 0.517 | 0.791 | 60.00 |
| [Z 296-22](../data/galaxy_sources/z_296_22.json) | spiral | 120.5979 | 388.078 | 0.02729 | 0.260 | 0.881 | 155.00 |
| [Z 264-17](../data/galaxy_sources/z_264_17.json) | lenticular | 120.5979 | 388.078 | 0.02729 | 0.540 | 0.453 | 26.00 |
| [IC 3002](../data/galaxy_sources/ic_3002.json) | spiral | 120.5979 | 388.078 | 0.02729 | 0.578 | 0.847 | 85.00 |
| [MCG+09-19-062](../data/galaxy_sources/mcg_09_19_062.json) | spiral | 120.6418 | 388.217 | 0.0273 | 1.025 | 0.373 | 97.00 |
| [MCG+07-19-033](../data/galaxy_sources/mcg_07_19_033.json) | barred spiral | 120.6418 | 388.217 | 0.0273 | 0.631 | 0.859 | 23.00 |
| [LEDA 28292](../data/galaxy_sources/leda_28292.json) | spiral | 120.6418 | 388.217 | 0.0273 | 0.408 | 0.528 | 142.00 |
| [LEDA 984709](../data/galaxy_sources/leda_984709.json) | lenticular | 120.6857 | 388.357 | 0.02731 | 0.546 | 0.674 | 157.00 |
| [UGC 12866](../data/galaxy_sources/ugc_12866.json) | spiral | 120.6921 | 388.377 | 0.02731146 | 0.782 | 0.287 | 141.00 |
| [IC 2928](../data/galaxy_sources/ic_2928.json) | spiral | 120.7072 | 388.425 | 0.02731489 | 0.872 | 0.855 | 136.00 |
| [Z 51-25](../data/galaxy_sources/z_51_25.json) | spiral | 120.7296 | 388.496 | 0.02732 | 0.623 | 0.883 | 110.00 |
| [MCG+00-19-032](../data/galaxy_sources/mcg_00_19_032.json) | elliptical | 120.7735 | 388.635 | 0.02733 | 0.590 | 0.645 | 93.00 |
| [Z 182-48](../data/galaxy_sources/z_182_48.json) | elliptical | 120.7735 | 388.635 | 0.02733 | 0.737 | 0.615 | 75.00 |
| [LEDA 28691](../data/galaxy_sources/leda_28691.json) | spiral | 120.7735 | 388.635 | 0.02733 | 0.854 | 0.208 | 45.00 |
| [IC 3001](../data/galaxy_sources/ic_3001.json) | barred spiral | 120.7735 | 388.635 | 0.02733 | 0.875 | 0.494 | 69.00 |
| [Z 191-53](../data/galaxy_sources/z_191_53.json) | spiral | 120.8126 | 388.759 | 0.02733889 | 1.025 | 0.863 | 111.00 |
| [Z 208-2](../data/galaxy_sources/z_208_2.json) | spiral | 120.8174 | 388.775 | 0.02734 | 0.517 | 0.610 | 2.00 |
| [MCG+08-24-015](../data/galaxy_sources/mcg_08_24_015.json) | barred spiral | 120.8174 | 388.775 | 0.02734 | 0.486 | 0.736 | 34.00 |
| [Z 91-4](../data/galaxy_sources/z_91_4.json) | barred spiral | 120.8174 | 388.775 | 0.02734 | 0.674 | 0.758 | 175.00 |
| [Z 48-58](../data/galaxy_sources/z_48_58.json) | spiral | 120.8174 | 388.775 | 0.02734 | 0.636 | 0.271 | 177.00 |
| [NGC 6319](../data/galaxy_sources/ngc_6319.json) | elliptical | 120.8613 | 388.914 | 0.02735 | 1.122 | 0.916 | 145.00 |
| [MCG+07-19-006](../data/galaxy_sources/mcg_07_19_006.json) | spiral | 120.8613 | 388.914 | 0.02735 | 0.740 | 0.376 | 25.00 |
| [Z 20-6](../data/galaxy_sources/z_20_6.json) | barred spiral | 120.8613 | 388.914 | 0.02735 | 0.403 | 0.766 | 117.00 |
| [LEDA 1129823](../data/galaxy_sources/leda_1129823.json) | spiral | 120.8613 | 388.914 | 0.02735 | 0.578 | 0.176 | 149.00 |
| [LEDA 93853](../data/galaxy_sources/leda_93853.json) | spiral | 120.8728 | 388.950 | 0.02735261 | 0.183 | 0.902 | 65.00 |
| [Z 148-69](../data/galaxy_sources/z_148_69.json) | lenticular | 120.9052 | 389.053 | 0.02736 | 0.858 | 0.488 | 173.00 |
| [MCG+10-24-122](../data/galaxy_sources/mcg_10_24_122.json) | spiral | 120.9491 | 389.193 | 0.02737 | 0.727 | 0.331 | 146.00 |
| [LEDA 79081](../data/galaxy_sources/leda_79081.json) | spiral | 120.9491 | 389.193 | 0.02737 | 0.298 | 0.731 | 149.00 |
| [LEDA 23434](../data/galaxy_sources/leda_23434.json) | spiral | 120.9930 | 389.332 | 0.02738 | 0.664 | 0.327 | 35.00 |
| [LEDA 86455](../data/galaxy_sources/leda_86455.json) | spiral | 120.9930 | 389.332 | 0.02738 | 0.473 | 0.796 | 56.00 |
| [IC 5177](../data/galaxy_sources/ic_5177.json) | spiral | 121.0233 | 389.428 | 0.0273869 | 0.792 | 0.802 | 11.00 |
| [Z 12-95](../data/galaxy_sources/z_12_95.json) | lenticular | 121.0369 | 389.472 | 0.02739 | 0.855 | 0.639 | 105.00 |
| [MCG+07-28-018](../data/galaxy_sources/mcg_07_28_018.json) | spiral | 121.0369 | 389.472 | 0.02739 | 0.612 | 0.580 | 168.00 |
| [LEDA 60041](../data/galaxy_sources/leda_60041.json) | barred spiral | 121.1247 | 389.750 | 0.02741 | 0.500 | 0.926 | 56.00 |
| [Z 294-23](../data/galaxy_sources/z_294_23.json) | lenticular | 121.1247 | 389.750 | 0.02741 | 0.642 | 0.722 | 73.00 |
| [Z 316-10](../data/galaxy_sources/z_316_10.json) | spiral | 121.1686 | 389.890 | 0.02742 | 0.563 | 0.899 | 78.00 |
| [Z 33-35](../data/galaxy_sources/z_33_35.json) | spiral | 121.2125 | 390.029 | 0.02743 | 0.581 | 0.341 | 57.00 |
| [LEDA 86444](../data/galaxy_sources/leda_86444.json) | spiral | 121.2125 | 390.029 | 0.02743 | 0.560 | 0.852 | 178.00 |
| [Z 476-68](../data/galaxy_sources/z_476_68.json) | spiral | 121.2492 | 390.145 | 0.02743835 | 0.761 | 0.927 | 140.00 |
| [MCG+07-19-041](../data/galaxy_sources/mcg_07_19_041.json) | barred spiral | 121.2564 | 390.168 | 0.02744 | 1.437 | 0.175 | 12.00 |
| [Z 47-53](../data/galaxy_sources/z_47_53.json) | barred spiral | 121.2564 | 390.168 | 0.02744 | 0.866 | 0.667 | 169.00 |
| [Z 461-26](../data/galaxy_sources/z_461_26.json) | barred spiral | 121.2943 | 390.289 | 0.02744864 | 0.593 | 0.857 | 169.00 |
| [MCG+10-21-003](../data/galaxy_sources/mcg_10_21_003.json) | spiral | 121.3003 | 390.308 | 0.02745 | 0.602 | 0.559 | 117.00 |
| [Z 380-2](../data/galaxy_sources/z_380_2.json) | spiral | 121.3003 | 390.308 | 0.02745 | 0.495 | 0.871 | 22.00 |
| [LEDA 35716](../data/galaxy_sources/leda_35716.json) | spiral | 121.3442 | 390.447 | 0.02746 | 0.354 | 0.950 | 51.00 |
| [IC 1054](../data/galaxy_sources/ic_1054.json) | lenticular | 121.3442 | 390.447 | 0.02746 | 1.205 | 0.725 | 9.00 |
| [Z 415-33](../data/galaxy_sources/z_415_33.json) | spiral | 121.3847 | 390.575 | 0.02746922 | 0.612 | 0.783 | 50.00 |
| [ESO 405-34](../data/galaxy_sources/eso_405_34.json) | spiral | 121.3881 | 390.586 | 0.02747 | 0.220 | 0.718 | 105.00 |
| [UGC 6108](../data/galaxy_sources/ugc_6108.json) | spiral | 121.3881 | 390.586 | 0.02747 | 0.799 | 0.220 | 88.00 |
| [LEDA 1107110](../data/galaxy_sources/leda_1107110.json) | lenticular | 121.3881 | 390.586 | 0.02747 | 0.679 | 0.516 | 84.00 |
| [UGC 8075](../data/galaxy_sources/ugc_8075.json) | spiral | 121.3881 | 390.586 | 0.02747 | 1.459 | 0.144 | 131.00 |
| [Z 147-56](../data/galaxy_sources/z_147_56.json) | weakly barred spiral | 121.4298 | 390.719 | 0.02747951 | 1.572 | 0.192 | 134.00 |
| [Z 32-54](../data/galaxy_sources/z_32_54.json) | barred spiral | 121.4298 | 390.719 | 0.02747951 | 1.193 | 0.453 | 30.00 |
| [NGC 7211](../data/galaxy_sources/ngc_7211.json) | lenticular | 121.4320 | 390.726 | 0.02748 | 0.856 | 0.808 | 79.00 |
| [Z 264-87](../data/galaxy_sources/z_264_87.json) | spiral | 121.4759 | 390.865 | 0.02749 | 1.102 | 0.508 | 14.00 |
| [Z 238-57](../data/galaxy_sources/z_238_57.json) | spiral | 121.4759 | 390.865 | 0.02749 | 0.819 | 0.634 | 35.00 |
| [MCG+09-19-113](../data/galaxy_sources/mcg_09_19_113.json) | spiral | 121.4759 | 390.865 | 0.02749 | 1.505 | 0.139 | 173.00 |
| [Z 148-35](../data/galaxy_sources/z_148_35.json) | lenticular | 121.5198 | 391.004 | 0.0275 | 0.849 | 0.381 | 27.00 |
| [Z 180-40](../data/galaxy_sources/z_180_40.json) | spiral | 121.5637 | 391.144 | 0.02751 | 0.552 | 0.813 | 151.00 |
| [Mrk 112](../data/galaxy_sources/mrk_112.json) | elliptical | 121.6076 | 391.283 | 0.02752 | 0.535 | 0.819 | 93.00 |
| [MCG+08-25-033](../data/galaxy_sources/mcg_08_25_033.json) | spiral | 121.6076 | 391.283 | 0.02752 | 0.681 | 0.667 | 173.00 |
| [MCG+10-12-097](../data/galaxy_sources/mcg_10_12_097.json) | spiral | 121.6515 | 391.422 | 0.02753 | 0.554 | 0.655 | 75.00 |
| [Z 291-19](../data/galaxy_sources/z_291_19.json) | barred spiral | 121.6515 | 391.422 | 0.02753 | 0.655 | 0.967 | 32.00 |
| [MCG+10-12-113](../data/galaxy_sources/mcg_10_12_113.json) | spiral | 121.6954 | 391.562 | 0.02754 | 0.533 | 0.816 | 174.00 |
| [Z 103-17](../data/galaxy_sources/z_103_17.json) | spiral | 121.6954 | 391.562 | 0.02754 | 1.309 | 0.125 | 135.00 |
| [LEDA 84871](../data/galaxy_sources/leda_84871.json) | spiral | 121.7393 | 391.701 | 0.02755 | 0.542 | 0.522 | 30.00 |
| [Z 148-45](../data/galaxy_sources/z_148_45.json) | lenticular | 121.7393 | 391.701 | 0.02755 | 0.661 | 0.589 | 88.00 |
| [Z 238-31](../data/galaxy_sources/z_238_31.json) | spiral | 121.7832 | 391.840 | 0.02756 | 0.742 | 0.836 | 174.00 |
| [LEDA 191324](../data/galaxy_sources/leda_191324.json) | lenticular | 121.7832 | 391.840 | 0.02756 | 0.513 | 0.707 | 98.00 |
| [Z 150-12](../data/galaxy_sources/z_150_12.json) | spiral | 121.8213 | 391.961 | 0.02756869 | 0.520 | 0.953 | 145.00 |
| [MCG+10-25-005](../data/galaxy_sources/mcg_10_25_005.json) | spiral | 121.8271 | 391.980 | 0.02757 | 0.706 | 0.354 | 135.00 |
| [LEDA 84872](../data/galaxy_sources/leda_84872.json) | spiral | 121.8271 | 391.980 | 0.02757 | 0.163 | 1.000 | 90.00 |
| [IC 4193](../data/galaxy_sources/ic_4193.json) | spiral | 121.8710 | 392.119 | 0.02758 | 0.529 | 0.692 | 62.00 |
| [IC 5241](../data/galaxy_sources/ic_5241.json) | barred spiral | 121.9267 | 392.296 | 0.0275927 | 0.771 | 0.898 | 6.00 |
| [Z 218-8](../data/galaxy_sources/z_218_8.json) | elliptical | 121.9588 | 392.398 | 0.0276 | 0.663 | 0.885 | 104.00 |
| [Z 102-31](../data/galaxy_sources/z_102_31.json) | spiral | 121.9869 | 392.487 | 0.02760642 | 0.549 | 0.779 | 82.00 |
| [Z 461-33](../data/galaxy_sources/z_461_33.json) | spiral | 122.0020 | 392.535 | 0.02760985 | 1.091 | 0.301 | 112.00 |
| [NGC 3410](../data/galaxy_sources/ngc_3410.json) | spiral | 122.0026 | 392.537 | 0.02761 | 0.834 | 0.721 | 32.00 |
| [LEDA 3083147](../data/galaxy_sources/leda_3083147.json) | lenticular | 122.0026 | 392.537 | 0.02761 | 0.594 | 0.495 | 52.00 |
| [LEDA 90994](../data/galaxy_sources/leda_90994.json) | spiral | 122.0465 | 392.676 | 0.02762 | 0.941 | 0.176 | 162.00 |
| [MCG+00-19-018](../data/galaxy_sources/mcg_00_19_018.json) | lenticular | 122.0904 | 392.816 | 0.02763 | 0.676 | 0.625 | 115.00 |
| [Z 91-5](../data/galaxy_sources/z_91_5.json) | barred spiral | 122.0904 | 392.816 | 0.02763 | 0.602 | 0.932 | 5.00 |
| [Z 438-47](../data/galaxy_sources/z_438_47.json) | spiral | 122.0923 | 392.822 | 0.02763043 | 0.625 | 0.577 | 162.00 |
| [LEDA 91799](../data/galaxy_sources/leda_91799.json) | spiral | 122.1224 | 392.917 | 0.02763729 | 1.335 | 0.140 | 51.00 |
| [Z 218-17](../data/galaxy_sources/z_218_17.json) | spiral | 122.1343 | 392.955 | 0.02764 | 0.680 | 0.854 | 9.00 |
| [Z 502-21](../data/galaxy_sources/z_502_21.json) | spiral | 122.1526 | 393.013 | 0.02764415 | 0.581 | 0.680 | 48.00 |
| [Z 75-68](../data/galaxy_sources/z_75_68.json) | spiral | 122.1782 | 393.094 | 0.02765 | 0.518 | 0.897 | 77.00 |
| [NGC 5123](../data/galaxy_sources/ngc_5123.json) | spiral | 122.2221 | 393.234 | 0.02766 | 1.126 | 0.970 | 163.00 |
| [LEDA 30963](../data/galaxy_sources/leda_30963.json) | spiral | 122.2221 | 393.234 | 0.02766 | 0.388 | 0.590 | 94.00 |
| [Z 427-30](../data/galaxy_sources/z_427_30.json) | spiral | 122.2660 | 393.373 | 0.02767 | 0.612 | 0.591 | 39.00 |
| [MCG+07-27-044](../data/galaxy_sources/mcg_07_27_044.json) | spiral | 122.2660 | 393.373 | 0.02767 | 0.613 | 0.695 | 59.00 |
| [Z 48-14](../data/galaxy_sources/z_48_14.json) | lenticular | 122.2660 | 393.373 | 0.02767 | 0.623 | 0.595 | 76.00 |
| [Z 198-14](../data/galaxy_sources/z_198_14.json) | spiral | 122.2660 | 393.373 | 0.02767 | 0.805 | 0.670 | 33.00 |
| [MCG+08-24-088](../data/galaxy_sources/mcg_08_24_088.json) | spiral | 122.3099 | 393.512 | 0.02768 | 0.620 | 0.746 | 113.00 |
| [LEDA 163048](../data/galaxy_sources/leda_163048.json) | spiral | 122.3099 | 393.512 | 0.02768 | 0.734 | 0.741 | 137.00 |
| [UGC 7770](../data/galaxy_sources/ugc_7770.json) | spiral | 122.3099 | 393.512 | 0.02768 | 1.312 | 0.170 | 151.00 |
| [LEDA 82557](../data/galaxy_sources/leda_82557.json) | barred spiral | 122.3977 | 393.791 | 0.0277 | 0.422 | 0.729 | 168.00 |
| [Z 103-77](../data/galaxy_sources/z_103_77.json) | barred spiral | 122.3977 | 393.791 | 0.0277 | 0.504 | 0.666 | 103.00 |
| [LEDA 1013346](../data/galaxy_sources/leda_1013346.json) | spiral | 122.3977 | 393.791 | 0.0277 | 0.718 | 0.402 | 20.00 |
| [LEDA 27315](../data/galaxy_sources/leda_27315.json) | spiral | 122.4416 | 393.930 | 0.02771 | 0.492 | 0.387 | 57.00 |
| [IC 852](../data/galaxy_sources/ic_852.json) | elliptical | 122.5294 | 394.209 | 0.02773 | 1.099 | 0.759 | 26.00 |
| [Z 182-53](../data/galaxy_sources/z_182_53.json) | spiral | 122.5733 | 394.348 | 0.02774 | 0.608 | 0.948 | 129.00 |
| [Z 413-61](../data/galaxy_sources/z_413_61.json) | barred spiral | 122.5742 | 394.351 | 0.02774021 | 0.447 | 0.807 | 160.00 |
| [Z 212-20](../data/galaxy_sources/z_212_20.json) | spiral | 122.6172 | 394.487 | 0.02775 | 0.851 | 0.495 | 49.00 |
| [LEDA 71247](../data/galaxy_sources/leda_71247.json) | spiral | 122.6193 | 394.494 | 0.0277505 | 0.491 | 0.790 | 30.00 |
| [Z 499-58](../data/galaxy_sources/z_499_58.json) | spiral | 122.6193 | 394.494 | 0.0277505 | 1.131 | 0.265 | 113.00 |
| [Z 76-146](../data/galaxy_sources/z_76_146.json) | spiral | 122.6610 | 394.627 | 0.02776 | 0.874 | 0.944 | 114.00 |
| [Z 209-52](../data/galaxy_sources/z_209_52.json) | spiral | 122.7049 | 394.766 | 0.02777 | 0.690 | 0.490 | 116.00 |
| [LEDA 12061](../data/galaxy_sources/leda_12061.json) | spiral | 122.7049 | 394.766 | 0.02777 | 0.495 | 0.410 | 73.00 |
| [UGC 1732](../data/galaxy_sources/ugc_1732.json) | spiral | 122.7097 | 394.781 | 0.02777108 | 1.091 | 0.141 | 24.00 |
| [Z 247-33](../data/galaxy_sources/z_247_33.json) | spiral | 122.7488 | 394.905 | 0.02778 | 1.089 | 0.459 | 45.00 |
| [Mrk 793](../data/galaxy_sources/mrk_793.json) | spiral | 122.7927 | 395.044 | 0.02779 | 0.441 | 0.926 | 158.00 |
| [Z 463-51](../data/galaxy_sources/z_463_51.json) | barred spiral | 122.8151 | 395.115 | 0.0277951 | 0.577 | 0.839 | 85.00 |
| [MCG+02-55-019](../data/galaxy_sources/mcg_02_55_019.json) | lenticular | 122.8366 | 395.184 | 0.0278 | 0.423 | 0.861 | 125.00 |
| [LEDA 1040099](../data/galaxy_sources/leda_1040099.json) | lenticular | 122.8366 | 395.184 | 0.0278 | 0.456 | 0.965 | 107.00 |
| [Z 162-68](../data/galaxy_sources/z_162_68.json) | spiral | 122.8753 | 395.307 | 0.02780882 | 0.434 | 0.816 | 115.00 |
| [Z 389-19](../data/galaxy_sources/z_389_19.json) | barred spiral | 122.8805 | 395.323 | 0.02781 | 1.125 | 0.449 | 21.00 |
| [Z 20-28](../data/galaxy_sources/z_20_28.json) | spiral | 122.8805 | 395.323 | 0.02781 | 0.519 | 0.555 | 76.00 |
| [Z 460-51](../data/galaxy_sources/z_460_51.json) | spiral | 122.9506 | 395.546 | 0.02782598 | 0.607 | 0.452 | 145.00 |
| [Z 235-42](../data/galaxy_sources/z_235_42.json) | spiral | 122.9683 | 395.602 | 0.02783 | 0.622 | 0.604 | 147.00 |
| [LEDA 1028667](../data/galaxy_sources/leda_1028667.json) | lenticular | 122.9683 | 395.602 | 0.02783 | 0.581 | 0.750 | 115.00 |
| [LEDA 145695](../data/galaxy_sources/leda_145695.json) | elliptical | 123.0122 | 395.741 | 0.02784 | 0.714 | 0.957 | 173.00 |
| [Z 92-36](../data/galaxy_sources/z_92_36.json) | elliptical | 123.0122 | 395.741 | 0.02784 | 0.757 | 0.723 | 93.00 |
| [Z 155-83](../data/galaxy_sources/z_155_83.json) | elliptical | 123.0122 | 395.741 | 0.02784 | 0.904 | 0.878 | 126.00 |
| [MCG+08-25-037](../data/galaxy_sources/mcg_08_25_037.json) | spiral | 123.0561 | 395.880 | 0.02785 | 0.430 | 0.935 | 117.00 |
| [Z 148-39](../data/galaxy_sources/z_148_39.json) | spiral | 123.0999 | 396.019 | 0.02786 | 0.705 | 0.511 | 81.00 |
| [Z 119-115](../data/galaxy_sources/z_119_115.json) | spiral | 123.0999 | 396.019 | 0.02786 | 0.764 | 0.376 | 103.00 |
| [Z 37-100](../data/galaxy_sources/z_37_100.json) | spiral | 123.0999 | 396.019 | 0.02786 | 1.269 | 0.609 | 7.00 |
| [LEDA 34339](../data/galaxy_sources/leda_34339.json) | elliptical | 123.1438 | 396.159 | 0.02787 | 0.493 | 0.657 | 115.00 |
| [Z 484-16](../data/galaxy_sources/z_484_16.json) | spiral | 123.1765 | 396.262 | 0.02787745 | 0.886 | 0.599 | 57.00 |
| [Z 246-1](../data/galaxy_sources/z_246_1.json) | elliptical | 123.2316 | 396.437 | 0.02789 | 0.662 | 0.904 | 66.00 |
| [Z 118-7](../data/galaxy_sources/z_118_7.json) | spiral | 123.2316 | 396.437 | 0.02789 | 0.505 | 0.859 | 129.00 |
| [Z 34-35](../data/galaxy_sources/z_34_35.json) | spiral | 123.2316 | 396.437 | 0.02789 | 0.723 | 0.487 | 152.00 |
| [NGC 7132](../data/galaxy_sources/ngc_7132.json) | spiral | 123.2669 | 396.549 | 0.02789803 | 0.863 | 0.802 | 115.00 |
| [Z 207-54](../data/galaxy_sources/z_207_54.json) | spiral | 123.2755 | 396.576 | 0.0279 | 0.717 | 0.823 | 55.00 |
| [Z 437-8](../data/galaxy_sources/z_437_8.json) | elliptical | 123.2755 | 396.576 | 0.0279 | 0.938 | 0.738 | 106.00 |
| [Z 155-64](../data/galaxy_sources/z_155_64.json) | spiral | 123.3194 | 396.716 | 0.02791 | 0.570 | 0.696 | 28.00 |
| [Z 427-19](../data/galaxy_sources/z_427_19.json) | spiral | 123.3271 | 396.740 | 0.02791176 | 0.683 | 0.576 | 94.00 |
| [Z 456-19](../data/galaxy_sources/z_456_19.json) | spiral | 123.3572 | 396.836 | 0.02791862 | 0.586 | 0.696 | 91.00 |
| [Z 91-23](../data/galaxy_sources/z_91_23.json) | spiral | 123.3723 | 396.883 | 0.02792205 | 1.268 | 0.224 | 58.00 |
| [LEDA 84057](../data/galaxy_sources/leda_84057.json) | spiral | 123.4072 | 396.994 | 0.02793 | 0.632 | 0.352 | 108.00 |
| [Z 47-142](../data/galaxy_sources/z_47_142.json) | elliptical | 123.4072 | 396.994 | 0.02793 | 0.809 | 0.535 | 149.00 |
| [Z 461-44](../data/galaxy_sources/z_461_44.json) | spiral | 123.4476 | 397.122 | 0.02793921 | 0.647 | 0.592 | 24.00 |
| [MCG+11-16-009](../data/galaxy_sources/mcg_11_16_009.json) | spiral | 123.4511 | 397.133 | 0.02794 | 1.005 | 0.477 | 137.00 |
| [Z 34-43](../data/galaxy_sources/z_34_43.json) | spiral | 123.4511 | 397.133 | 0.02794 | 0.799 | 0.844 | 159.00 |
| [Z 473-11](../data/galaxy_sources/z_473_11.json) | weakly barred spiral | 123.4928 | 397.266 | 0.0279495 | 0.640 | 0.853 | 30.00 |
| [Z 220-57](../data/galaxy_sources/z_220_57.json) | spiral | 123.4949 | 397.273 | 0.02795 | 0.536 | 0.790 | 133.00 |
| [Z 37-88](../data/galaxy_sources/z_37_88.json) | spiral | 123.4949 | 397.273 | 0.02795 | 1.042 | 0.594 | 122.00 |
| [Z 241-19](../data/galaxy_sources/z_241_19.json) | barred spiral | 123.5388 | 397.412 | 0.02796 | 0.773 | 0.910 | 178.00 |
| [LEDA 1016461](../data/galaxy_sources/leda_1016461.json) | lenticular | 123.5388 | 397.412 | 0.02796 | 0.442 | 0.763 | 85.00 |
| [Z 43-39](../data/galaxy_sources/z_43_39.json) | spiral | 123.5827 | 397.551 | 0.02797 | 0.871 | 0.187 | 139.00 |
| [Z 48-40](../data/galaxy_sources/z_48_40.json) | spiral | 123.5827 | 397.551 | 0.02797 | 0.639 | 0.900 | 133.00 |
| [Z 415-31](../data/galaxy_sources/z_415_31.json) | spiral | 123.5982 | 397.600 | 0.02797353 | 1.467 | 0.192 | 158.00 |
| [Z 459-42](../data/galaxy_sources/z_459_42.json) | barred spiral | 123.5982 | 397.600 | 0.02797353 | 1.127 | 0.508 | 45.00 |
| [ESO 468-18](../data/galaxy_sources/eso_468_18.json) | spiral | 123.6266 | 397.690 | 0.02798 | 0.267 | 0.539 | 105.00 |
| [Z 132-74](../data/galaxy_sources/z_132_74.json) | spiral | 123.6266 | 397.690 | 0.02798 | 0.690 | 0.923 | 42.00 |
| [NGC 927](../data/galaxy_sources/ngc_927.json) | barred spiral | 123.6584 | 397.791 | 0.02798725 | 1.157 | 0.923 | 13.00 |
| [Z 185-67](../data/galaxy_sources/z_185_67.json) | barred spiral | 123.6661 | 397.816 | 0.027989 | 0.857 | 0.508 | 120.00 |
| [LEDA 82570](../data/galaxy_sources/leda_82570.json) | spiral | 123.6705 | 397.830 | 0.02799 | 0.638 | 0.336 | 32.00 |
| [Z 206-35](../data/galaxy_sources/z_206_35.json) | spiral | 123.6705 | 397.830 | 0.02799 | 0.956 | 0.256 | 9.00 |
| [Z 74-127](../data/galaxy_sources/z_74_127.json) | spiral | 123.6705 | 397.830 | 0.02799 | 0.843 | 0.321 | 87.00 |
| [LEDA 1150350](../data/galaxy_sources/leda_1150350.json) | lenticular | 123.7144 | 397.969 | 0.028 | 0.580 | 0.403 | 173.00 |
| [Z 294-15](../data/galaxy_sources/z_294_15.json) | elliptical | 123.7583 | 398.108 | 0.02801 | 0.641 | 0.671 | 15.00 |
| [MCG+09-15-106](../data/galaxy_sources/mcg_09_15_106.json) | spiral | 123.7583 | 398.108 | 0.02801 | 1.089 | 0.186 | 151.00 |
| [LEDA 84412](../data/galaxy_sources/leda_84412.json) | spiral | 123.7583 | 398.108 | 0.02801 | 0.587 | 0.290 | 86.00 |
| [MCG+07-19-038](../data/galaxy_sources/mcg_07_19_038.json) | spiral | 123.7583 | 398.108 | 0.02801 | 0.966 | 0.274 | 169.00 |
| [NPM1G +01.0318](../data/galaxy_sources/npm1g_01_0318.json) | lenticular | 123.7583 | 398.108 | 0.02801 | 0.424 | 0.565 | 162.00 |
| [Z 414-6](../data/galaxy_sources/z_414_6.json) | spiral | 123.8241 | 398.317 | 0.028025 | 1.120 | 0.239 | 49.00 |
| [Z 438-48](../data/galaxy_sources/z_438_48.json) | spiral | 123.8392 | 398.365 | 0.02802843 | 0.407 | 0.619 | 35.00 |
| [LEDA 78895](../data/galaxy_sources/leda_78895.json) | spiral | 123.8460 | 398.387 | 0.02803 | 0.665 | 0.248 | 23.00 |
| [Z 120-38](../data/galaxy_sources/z_120_38.json) | spiral | 123.8460 | 398.387 | 0.02803 | 0.662 | 0.547 | 117.00 |
| [Z 163-12](../data/galaxy_sources/z_163_12.json) | lenticular | 123.8460 | 398.387 | 0.02803 | 0.263 | 0.882 | 60.00 |
| [Z 287-48](../data/galaxy_sources/z_287_48.json) | spiral | 123.8899 | 398.526 | 0.02804 | 0.632 | 0.816 | 171.00 |
| [LEDA 2589593](../data/galaxy_sources/leda_2589593.json) | spiral | 123.8899 | 398.526 | 0.02804 | 0.410 | 0.877 | 75.00 |
| [MCG+08-20-030](../data/galaxy_sources/mcg_08_20_030.json) | spiral | 123.8899 | 398.526 | 0.02804 | 1.154 | 0.295 | 57.00 |
| [LEDA 27365](../data/galaxy_sources/leda_27365.json) | spiral | 123.9338 | 398.665 | 0.02805 | 0.586 | 0.760 | 165.00 |
| [LEDA 87004](../data/galaxy_sources/leda_87004.json) | spiral | 123.9338 | 398.665 | 0.02805 | 0.489 | 0.525 | 10.00 |
| [Z 132-29](../data/galaxy_sources/z_132_29.json) | spiral | 123.9338 | 398.665 | 0.02805 | 0.739 | 0.413 | 164.00 |
| [Z 432-21](../data/galaxy_sources/z_432_21.json) | barred spiral | 123.9747 | 398.795 | 0.02805932 | 1.003 | 0.618 | 141.00 |
| [UGC 994](../data/galaxy_sources/ugc_994.json) | spiral | 123.9747 | 398.795 | 0.02805932 | 0.784 | 0.466 | 20.00 |
| [Z 241-18](../data/galaxy_sources/z_241_18.json) | lenticular | 123.9777 | 398.804 | 0.02806 | 0.812 | 0.687 | 159.00 |
| [Z 75-92](../data/galaxy_sources/z_75_92.json) | spiral | 124.0216 | 398.944 | 0.02807 | 0.970 | 0.399 | 130.00 |
| [MCG+08-26-014](../data/galaxy_sources/mcg_08_26_014.json) | spiral | 124.0655 | 399.083 | 0.02808 | 0.773 | 0.438 | 91.00 |
| [Z 95-102](../data/galaxy_sources/z_95_102.json) | spiral | 124.0655 | 399.083 | 0.02808 | 0.717 | 0.443 | 151.00 |
| [Z 248-49](../data/galaxy_sources/z_248_49.json) | lenticular | 124.1093 | 399.222 | 0.02809 | 0.814 | 0.315 | 176.00 |
| [Z 291-4](../data/galaxy_sources/z_291_4.json) | spiral | 124.1093 | 399.222 | 0.02809 | 1.072 | 0.229 | 80.00 |
| [Z 191-76](../data/galaxy_sources/z_191_76.json) | elliptical | 124.1093 | 399.222 | 0.02809 | 0.732 | 0.881 | 77.00 |
| [Z 170-12](../data/galaxy_sources/z_170_12.json) | spiral | 124.1093 | 399.222 | 0.02809 | 0.857 | 0.454 | 169.00 |
| [Z 461-52](../data/galaxy_sources/z_461_52.json) | spiral | 124.1253 | 399.273 | 0.02809364 | 0.509 | 0.579 | 92.00 |
| [MCG+10-19-043](../data/galaxy_sources/mcg_10_19_043.json) | spiral | 124.1532 | 399.361 | 0.0281 | 0.831 | 0.255 | 126.00 |
| [MCG+07-18-014](../data/galaxy_sources/mcg_07_18_014.json) | spiral | 124.1532 | 399.361 | 0.0281 | 0.509 | 0.665 | 13.00 |
| [LEDA 24803](../data/galaxy_sources/leda_24803.json) | spiral | 124.1532 | 399.361 | 0.0281 | 0.479 | 0.304 | 5.00 |
| [Z 90-31](../data/galaxy_sources/z_90_31.json) | spiral | 124.1532 | 399.361 | 0.0281 | 0.935 | 0.335 | 164.00 |
| [IC 595](../data/galaxy_sources/ic_595.json) | lenticular | 124.1532 | 399.361 | 0.0281 | 0.818 | 0.714 | 170.00 |
| [Z 36-67](../data/galaxy_sources/z_36_67.json) | spiral | 124.1532 | 399.361 | 0.0281 | 0.441 | 0.956 | 25.00 |
| [Z 76-1](../data/galaxy_sources/z_76_1.json) | spiral | 124.1532 | 399.361 | 0.0281 | 0.602 | 0.728 | 163.00 |
| [LEDA 90091](../data/galaxy_sources/leda_90091.json) | barred spiral | 124.1971 | 399.500 | 0.02811 | 0.475 | 0.950 | 180.00 |
| [LEDA 90910](../data/galaxy_sources/leda_90910.json) | spiral | 124.1971 | 399.500 | 0.02811 | 0.729 | 0.167 | 136.00 |
| [IC 2420](../data/galaxy_sources/ic_2420.json) | elliptical | 124.1971 | 399.500 | 0.02811 | 0.859 | 0.936 | 132.00 |
| [Z 238-2](../data/galaxy_sources/z_238_2.json) | spiral | 124.2410 | 399.640 | 0.02812 | 0.683 | 0.827 | 32.00 |
| [Z 210-8](../data/galaxy_sources/z_210_8.json) | spiral | 124.2410 | 399.640 | 0.02812 | 0.694 | 0.420 | 159.00 |
| [Z 37-91](../data/galaxy_sources/z_37_91.json) | spiral | 124.2410 | 399.640 | 0.02812 | 0.390 | 0.875 | 154.00 |
| [Z 38-118](../data/galaxy_sources/z_38_118.json) | spiral | 124.2410 | 399.640 | 0.02812 | 0.684 | 0.602 | 140.00 |
| [Z 64-83](../data/galaxy_sources/z_64_83.json) | spiral | 124.2849 | 399.779 | 0.02813 | 0.799 | 0.786 | 4.00 |
| [UGC 2300](../data/galaxy_sources/ugc_2300.json) | spiral | 124.3060 | 399.846 | 0.02813482 | 0.417 | 0.600 | 95.00 |
| [IC 570](../data/galaxy_sources/ic_570.json) | spiral | 124.3288 | 399.918 | 0.02814 | 0.548 | 0.795 | 50.00 |
| [Z 132-57](../data/galaxy_sources/z_132_57.json) | spiral | 124.3288 | 399.918 | 0.02814 | 0.641 | 0.561 | 117.00 |
| [Z 290-59](../data/galaxy_sources/z_290_59.json) | spiral | 124.3727 | 400.057 | 0.02815 | 1.353 | 0.255 | 40.00 |
| [Z 33-24](../data/galaxy_sources/z_33_24.json) | elliptical | 124.3727 | 400.057 | 0.02815 | 0.708 | 0.788 | 85.00 |
| [LEDA 84410](../data/galaxy_sources/leda_84410.json) | spiral | 124.4165 | 400.196 | 0.02816 | 0.608 | 0.303 | 136.00 |
| [LEDA 26637](../data/galaxy_sources/leda_26637.json) | spiral | 124.5921 | 400.753 | 0.0282 | 0.513 | 0.559 | 46.00 |
| [Z 80-37](../data/galaxy_sources/z_80_37.json) | spiral | 124.5921 | 400.753 | 0.0282 | 0.603 | 0.974 | 88.00 |
| [NGC 5830](../data/galaxy_sources/ngc_5830.json) | spiral | 124.6359 | 400.892 | 0.02821 | 0.982 | 0.617 | 175.00 |
| [MCG+08-24-064](../data/galaxy_sources/mcg_08_24_064.json) | spiral | 124.6359 | 400.892 | 0.02821 | 0.786 | 0.331 | 26.00 |
| [Z 33-6](../data/galaxy_sources/z_33_6.json) | spiral | 124.6359 | 400.892 | 0.02821 | 0.383 | 0.857 | 105.00 |
| [Z 452-27](../data/galaxy_sources/z_452_27.json) | spiral | 124.6374 | 400.897 | 0.02821033 | 1.272 | 0.514 | 109.00 |
| [UGC 12432](../data/galaxy_sources/ugc_12432.json) | spiral | 124.6525 | 400.945 | 0.02821376 | 1.219 | 0.198 | 109.00 |
| [Z 152-61](../data/galaxy_sources/z_152_61.json) | elliptical | 124.6798 | 401.032 | 0.02822 | 0.846 | 0.609 | 38.00 |
| [Z 93-94](../data/galaxy_sources/z_93_94.json) | spiral | 124.6798 | 401.032 | 0.02822 | 0.785 | 0.791 | 67.00 |
| [MCG+06-32-009](../data/galaxy_sources/mcg_06_32_009.json) | lenticular | 124.6798 | 401.032 | 0.02822 | 0.486 | 0.743 | 118.00 |
| [NGC 5568](../data/galaxy_sources/ngc_5568.json) | spiral | 124.6798 | 401.032 | 0.02822 | 0.699 | 0.817 | 99.00 |
| [Z 427-38](../data/galaxy_sources/z_427_38.json) | spiral | 124.7278 | 401.184 | 0.02823093 | 0.816 | 0.756 | 157.00 |
| [Z 35-3](../data/galaxy_sources/z_35_3.json) | spiral | 124.7676 | 401.310 | 0.02824 | 0.535 | 0.908 | 122.00 |
| [MCG+09-21-087](../data/galaxy_sources/mcg_09_21_087.json) | spiral | 124.8115 | 401.449 | 0.02825 | 0.849 | 0.353 | 159.00 |
| [Z 92-10](../data/galaxy_sources/z_92_10.json) | spiral | 124.8115 | 401.449 | 0.02825 | 1.016 | 0.781 | 154.00 |
| [ESO 150-15](../data/galaxy_sources/eso_150_15.json) | spiral | 124.8377 | 401.533 | 0.02825598 | 0.850 | 0.917 | 50.00 |
| [Z 93-11](../data/galaxy_sources/z_93_11.json) | barred spiral | 124.8554 | 401.588 | 0.02826 | 1.410 | 0.349 | 95.00 |
| [LEDA 51411](../data/galaxy_sources/leda_51411.json) | spiral | 124.8554 | 401.588 | 0.02826 | 0.316 | 0.825 | 150.00 |
| [MCG+06-32-025](../data/galaxy_sources/mcg_06_32_025.json) | elliptical | 124.8554 | 401.588 | 0.02826 | 0.647 | 0.671 | 37.00 |
| [Z 47-92](../data/galaxy_sources/z_47_92.json) | spiral | 124.8554 | 401.588 | 0.02826 | 0.953 | 0.256 | 93.00 |
| [Z 170-25](../data/galaxy_sources/z_170_25.json) | spiral | 124.8554 | 401.588 | 0.02826 | 0.746 | 0.418 | 108.00 |
| [Z 52-38](../data/galaxy_sources/z_52_38.json) | spiral | 124.8633 | 401.614 | 0.02826182 | 1.106 | 0.383 | 170.00 |
| [LEDA 23813](../data/galaxy_sources/leda_23813.json) | spiral | 124.8992 | 401.728 | 0.02827 | 0.327 | 0.895 | 33.00 |
| [Z 64-78](../data/galaxy_sources/z_64_78.json) | elliptical | 124.9431 | 401.867 | 0.02828 | 0.690 | 0.668 | 99.00 |
| [Z 459-35](../data/galaxy_sources/z_459_35.json) | spiral | 124.9537 | 401.900 | 0.02828241 | 0.751 | 0.567 | 127.00 |
| [Z 269-51](../data/galaxy_sources/z_269_51.json) | spiral | 124.9870 | 402.006 | 0.02829 | 1.028 | 0.292 | 84.00 |
| [LEDA 37495](../data/galaxy_sources/leda_37495.json) | spiral | 124.9870 | 402.006 | 0.02829 | 0.741 | 0.248 | 55.00 |
| [MCG+08-24-076](../data/galaxy_sources/mcg_08_24_076.json) | lenticular | 125.0309 | 402.145 | 0.0283 | 0.578 | 0.742 | 9.00 |
| [Z 36-69](../data/galaxy_sources/z_36_69.json) | spiral | 125.0309 | 402.145 | 0.0283 | 0.580 | 0.451 | 83.00 |
| [LEDA 32066](../data/galaxy_sources/leda_32066.json) | spiral | 125.0309 | 402.145 | 0.0283 | 0.557 | 0.602 | 102.00 |
| [MCG+07-28-064](../data/galaxy_sources/mcg_07_28_064.json) | spiral | 125.0748 | 402.284 | 0.02831 | 0.603 | 0.731 | 37.00 |
| [Z 426-58](../data/galaxy_sources/z_426_58.json) | spiral | 125.0748 | 402.284 | 0.02831 | 0.524 | 0.620 | 105.00 |
| [MCG+10-25-062](../data/galaxy_sources/mcg_10_25_062.json) | spiral | 125.1186 | 402.424 | 0.02832 | 0.774 | 0.304 | 148.00 |
| [Z 300-46](../data/galaxy_sources/z_300_46.json) | spiral | 125.1186 | 402.424 | 0.02832 | 1.034 | 0.402 | 43.00 |
| [Z 180-34](../data/galaxy_sources/z_180_34.json) | spiral | 125.1186 | 402.424 | 0.02832 | 0.470 | 0.400 | 75.00 |
| [Z 101-39](../data/galaxy_sources/z_101_39.json) | spiral | 125.1186 | 402.424 | 0.02832 | 0.470 | 0.566 | 108.00 |
| [LEDA 3083130](../data/galaxy_sources/leda_3083130.json) | spiral | 125.1625 | 402.563 | 0.02833 | 0.457 | 0.677 | 16.00 |
| [Z 36-40](../data/galaxy_sources/z_36_40.json) | lenticular | 125.1625 | 402.563 | 0.02833 | 0.681 | 0.630 | 68.00 |
| [Z 12-52](../data/galaxy_sources/z_12_52.json) | barred spiral | 125.1625 | 402.563 | 0.02833 | 1.161 | 0.682 | 125.00 |
| [Z 61-28](../data/galaxy_sources/z_61_28.json) | lenticular | 125.2503 | 402.841 | 0.02835 | 0.393 | 0.965 | 21.00 |
| [MCG+06-32-094](../data/galaxy_sources/mcg_06_32_094.json) | spiral | 125.2503 | 402.841 | 0.02835 | 1.144 | 0.294 | 60.00 |
| [MCG+10-25-060](../data/galaxy_sources/mcg_10_25_060.json) | spiral | 125.2941 | 402.980 | 0.02836 | 1.147 | 0.202 | 142.00 |
| [MCG+10-25-067](../data/galaxy_sources/mcg_10_25_067.json) | spiral | 125.2941 | 402.980 | 0.02836 | 0.659 | 0.699 | 141.00 |
| [Z 191-66](../data/galaxy_sources/z_191_66.json) | spiral | 125.3303 | 403.095 | 0.02836823 | 0.678 | 0.685 | 68.00 |
| [LEDA 3092489](../data/galaxy_sources/leda_3092489.json) | spiral | 125.3380 | 403.119 | 0.02837 | 0.223 | 0.861 | 15.00 |
| [LEDA 1027819](../data/galaxy_sources/leda_1027819.json) | spiral | 125.3380 | 403.119 | 0.02837 | 0.644 | 0.279 | 45.00 |
| [Z 63-3](../data/galaxy_sources/z_63_3.json) | elliptical | 125.3819 | 403.259 | 0.02838 | 0.373 | 0.780 | 135.00 |
| [Z 269-34](../data/galaxy_sources/z_269_34.json) | barred spiral | 125.4697 | 403.537 | 0.0284 | 1.484 | 0.404 | 46.00 |
| [MCG+06-17-027](../data/galaxy_sources/mcg_06_17_027.json) | spiral | 125.4697 | 403.537 | 0.0284 | 0.715 | 0.306 | 143.00 |
| [LEDA 79310](../data/galaxy_sources/leda_79310.json) | spiral | 125.4697 | 403.537 | 0.0284 | 0.220 | 0.741 | 115.00 |
| [Z 6-25](../data/galaxy_sources/z_6_25.json) | spiral | 125.4697 | 403.537 | 0.0284 | 0.786 | 0.297 | 86.00 |
| [MCG+06-31-088](../data/galaxy_sources/mcg_06_31_088.json) | spiral | 125.4697 | 403.537 | 0.0284 | 0.873 | 0.261 | 76.00 |
| [LEDA 1033028](../data/galaxy_sources/leda_1033028.json) | lenticular | 125.4697 | 403.537 | 0.0284 | 0.500 | 0.609 | 36.00 |
| [MCG+07-19-001](../data/galaxy_sources/mcg_07_19_001.json) | spiral | 125.5135 | 403.676 | 0.02841 | 0.818 | 0.317 | 17.00 |
| [LEDA 27599](../data/galaxy_sources/leda_27599.json) | spiral | 125.5135 | 403.676 | 0.02841 | 0.526 | 0.679 | 56.00 |
| [LEDA 1159724](../data/galaxy_sources/leda_1159724.json) | spiral | 125.5574 | 403.815 | 0.02842 | 0.763 | 0.246 | 20.00 |
| [LEDA 3104458](../data/galaxy_sources/leda_3104458.json) | spiral | 125.5574 | 403.815 | 0.02842 | 0.661 | 0.461 | 169.00 |
| [MCG+09-26-024](../data/galaxy_sources/mcg_09_26_024.json) | spiral | 125.6013 | 403.954 | 0.02843 | 0.637 | 0.704 | 160.00 |
| [Z 221-7](../data/galaxy_sources/z_221_7.json) | spiral | 125.6452 | 404.093 | 0.02844 | 0.761 | 0.411 | 19.00 |
| [Z 218-57](../data/galaxy_sources/z_218_57.json) | spiral | 125.6452 | 404.093 | 0.02844 | 0.592 | 0.860 | 81.00 |
| [MCG-01-53-023](../data/galaxy_sources/mcg_01_53_023.json) | spiral | 125.6452 | 404.093 | 0.02844 | 1.395 | 0.229 | 18.00 |
| [MCG+09-21-076](../data/galaxy_sources/mcg_09_21_076.json) | spiral | 125.6890 | 404.233 | 0.02845 | 0.722 | 0.518 | 153.00 |
| [MCG+08-24-010](../data/galaxy_sources/mcg_08_24_010.json) | spiral | 125.6890 | 404.233 | 0.02845 | 0.654 | 0.882 | 16.00 |
| [Z 122-58](../data/galaxy_sources/z_122_58.json) | spiral | 125.6890 | 404.233 | 0.02845 | 0.601 | 0.771 | 49.00 |
| [Z 241-96](../data/galaxy_sources/z_241_96.json) | spiral | 125.7329 | 404.372 | 0.02846 | 0.771 | 0.614 | 141.00 |
| [LEDA 132544](../data/galaxy_sources/leda_132544.json) | lenticular | 125.7329 | 404.372 | 0.02846 | 0.453 | 0.600 | 35.00 |
| [Z 33-7](../data/galaxy_sources/z_33_7.json) | spiral | 125.7768 | 404.511 | 0.02847 | 0.719 | 0.750 | 133.00 |
| [Z 91-90](../data/galaxy_sources/z_91_90.json) | spiral | 125.7768 | 404.511 | 0.02847 | 0.573 | 0.549 | 138.00 |
| [LEDA 89650](../data/galaxy_sources/leda_89650.json) | spiral | 125.7768 | 404.511 | 0.02847 | 0.428 | 0.365 | 150.00 |
| [Z 403-29](../data/galaxy_sources/z_403_29.json) | spiral | 125.7973 | 404.576 | 0.02847466 | 0.993 | 0.458 | 126.00 |
| [MCG+11-11-042](../data/galaxy_sources/mcg_11_11_042.json) | barred spiral | 125.8646 | 404.789 | 0.02849 | 0.452 | 0.633 | 50.00 |
| [MCG+08-24-110](../data/galaxy_sources/mcg_08_24_110.json) | spiral | 125.9084 | 404.928 | 0.0285 | 0.497 | 0.818 | 21.00 |
| [NGC 7085](../data/galaxy_sources/ngc_7085.json) | barred spiral | 125.9084 | 404.928 | 0.0285 | 1.218 | 0.466 | 148.00 |
| [IC 2615](../data/galaxy_sources/ic_2615.json) | spiral | 125.9084 | 404.928 | 0.0285 | 0.577 | 0.898 | 83.00 |
| [Z 118-2](../data/galaxy_sources/z_118_2.json) | lenticular | 125.9523 | 405.067 | 0.02851 | 0.273 | 0.582 | 155.00 |
| [LEDA 51954](../data/galaxy_sources/leda_51954.json) | spiral | 125.9962 | 405.207 | 0.02852 | 0.553 | 0.442 | 114.00 |
| [Z 380-57](../data/galaxy_sources/z_380_57.json) | spiral | 126.0401 | 405.346 | 0.02853 | 0.603 | 0.458 | 104.00 |
| [Z 118-8](../data/galaxy_sources/z_118_8.json) | lenticular | 126.0839 | 405.485 | 0.02854 | 0.660 | 0.897 | 88.00 |
| [Z 17-63](../data/galaxy_sources/z_17_63.json) | spiral | 126.2156 | 405.902 | 0.02857 | 0.885 | 0.783 | 49.00 |
| [Z 80-11](../data/galaxy_sources/z_80_11.json) | spiral | 126.2156 | 405.902 | 0.02857 | 0.788 | 0.428 | 57.00 |
| [Z 264-80](../data/galaxy_sources/z_264_80.json) | spiral | 126.2594 | 406.041 | 0.02858 | 0.528 | 0.536 | 132.00 |
| [Z 93-70](../data/galaxy_sources/z_93_70.json) | spiral | 126.2594 | 406.041 | 0.02858 | 0.513 | 0.942 | 161.00 |
| [MCG+09-23-006](../data/galaxy_sources/mcg_09_23_006.json) | spiral | 126.3033 | 406.180 | 0.02859 | 0.745 | 0.259 | 144.00 |
| [Z 47-120](../data/galaxy_sources/z_47_120.json) | barred spiral | 126.3033 | 406.180 | 0.02859 | 0.703 | 0.817 | 128.00 |
| [Z 249-9](../data/galaxy_sources/z_249_9.json) | spiral | 126.3472 | 406.320 | 0.0286 | 1.033 | 0.710 | 52.00 |
| [Z 91-87](../data/galaxy_sources/z_91_87.json) | barred spiral | 126.3472 | 406.320 | 0.0286 | 0.580 | 0.716 | 69.00 |
| [Z 122-97](../data/galaxy_sources/z_122_97.json) | barred spiral | 126.3472 | 406.320 | 0.0286 | 1.251 | 0.462 | 60.00 |
| [NGC 5509](../data/galaxy_sources/ngc_5509.json) | elliptical | 126.3472 | 406.320 | 0.0286 | 1.210 | 0.607 | 93.00 |
| [LEDA 1104081](../data/galaxy_sources/leda_1104081.json) | elliptical | 126.3472 | 406.320 | 0.0286 | 0.147 | 1.000 | 90.00 |
| [Z 92-29](../data/galaxy_sources/z_92_29.json) | spiral | 126.3910 | 406.459 | 0.02861 | 0.492 | 0.927 | 170.00 |
| [Z 63-18](../data/galaxy_sources/z_63_18.json) | spiral | 126.3910 | 406.459 | 0.02861 | 1.267 | 0.174 | 131.00 |
| [Z 123-15](../data/galaxy_sources/z_123_15.json) | spiral | 126.3910 | 406.459 | 0.02861 | 0.715 | 0.437 | 134.00 |
| [LEDA 85368](../data/galaxy_sources/leda_85368.json) | spiral | 126.4556 | 406.663 | 0.02862471 | 0.615 | 0.647 | 15.00 |
| [Z 461-41](../data/galaxy_sources/z_461_41.json) | barred spiral | 126.4752 | 406.725 | 0.02862917 | 0.617 | 0.480 | 100.00 |
| [Z 264-55](../data/galaxy_sources/z_264_55.json) | spiral | 126.5227 | 406.876 | 0.02864 | 0.512 | 0.521 | 79.00 |
| [Z 182-29](../data/galaxy_sources/z_182_29.json) | spiral | 126.5665 | 407.015 | 0.02865 | 0.523 | 0.459 | 75.00 |
| [Z 123-19](../data/galaxy_sources/z_123_19.json) | spiral | 126.5665 | 407.015 | 0.02865 | 0.639 | 0.370 | 150.00 |
| [Z 12-27](../data/galaxy_sources/z_12_27.json) | spiral | 126.6104 | 407.154 | 0.02866 | 0.550 | 0.718 | 134.00 |
| [ZW VIII 416](../data/galaxy_sources/zw_viii_416.json) | barred spiral | 126.6104 | 407.154 | 0.02866 | 0.413 | 0.880 | 16.00 |
| [Z 404-12](../data/galaxy_sources/z_404_12.json) | barred spiral | 126.6108 | 407.155 | 0.02866008 | 0.818 | 0.712 | 149.00 |
| [MCG+10-21-030](../data/galaxy_sources/mcg_10_21_030.json) | spiral | 126.6543 | 407.293 | 0.02867 | 0.753 | 0.295 | 159.00 |
| [Z 63-67](../data/galaxy_sources/z_63_67.json) | spiral | 126.6543 | 407.293 | 0.02867 | 0.539 | 0.472 | 45.00 |
| [Z 39-200](../data/galaxy_sources/z_39_200.json) | weakly barred spiral | 126.6543 | 407.293 | 0.02867 | 0.865 | 0.340 | 77.00 |
| [Z 21-80](../data/galaxy_sources/z_21_80.json) | spiral | 126.6543 | 407.293 | 0.02867 | 0.858 | 0.331 | 173.00 |
| [Z 33-8](../data/galaxy_sources/z_33_8.json) | spiral | 126.6982 | 407.432 | 0.02868 | 0.956 | 0.281 | 156.00 |
| [Z 63-30](../data/galaxy_sources/z_63_30.json) | spiral | 126.6982 | 407.432 | 0.02868 | 0.952 | 0.222 | 62.00 |
| [Z 91-81](../data/galaxy_sources/z_91_81.json) | spiral | 126.7420 | 407.572 | 0.02869 | 1.506 | 0.144 | 93.00 |
| [Z 12-53](../data/galaxy_sources/z_12_53.json) | spiral | 126.7420 | 407.572 | 0.02869 | 0.853 | 0.613 | 44.00 |
| [LEDA 3105771](../data/galaxy_sources/leda_3105771.json) | elliptical | 126.7420 | 407.572 | 0.02869 | 0.350 | 0.820 | 25.00 |
| [IC 3656](../data/galaxy_sources/ic_3656.json) | spiral | 126.7859 | 407.711 | 0.0287 | 0.383 | 0.520 | 100.00 |
| [Z 278-18](../data/galaxy_sources/z_278_18.json) | spiral | 126.8298 | 407.850 | 0.02871 | 0.785 | 0.719 | 164.00 |
| [Z 264-42](../data/galaxy_sources/z_264_42.json) | spiral | 126.8298 | 407.850 | 0.02871 | 0.416 | 0.812 | 15.00 |
| [Z 120-42](../data/galaxy_sources/z_120_42.json) | barred spiral | 126.8298 | 407.850 | 0.02871 | 0.441 | 0.766 | 2.00 |
| [NGC 7773](../data/galaxy_sources/ngc_7773.json) | barred spiral | 126.8518 | 407.920 | 0.02871502 | 1.160 | 0.939 | 133.00 |
| [Z 72-19](../data/galaxy_sources/z_72_19.json) | spiral | 126.8736 | 407.989 | 0.02872 | 0.447 | 0.705 | 170.00 |
| [Z 460-7](../data/galaxy_sources/z_460_7.json) | spiral | 126.9121 | 408.111 | 0.02872876 | 0.907 | 0.445 | 56.00 |
| [Z 247-37](../data/galaxy_sources/z_247_37.json) | elliptical | 126.9175 | 408.128 | 0.02873 | 0.647 | 0.829 | 89.00 |
| [Z 20-11](../data/galaxy_sources/z_20_11.json) | spiral | 126.9175 | 408.128 | 0.02873 | 0.962 | 0.396 | 41.00 |
| [Z 12-99](../data/galaxy_sources/z_12_99.json) | spiral | 126.9614 | 408.267 | 0.02874 | 0.475 | 0.771 | 176.00 |
| [Z 75-114](../data/galaxy_sources/z_75_114.json) | spiral | 126.9614 | 408.267 | 0.02874 | 1.365 | 0.164 | 64.00 |
| [LEDA 35597](../data/galaxy_sources/leda_35597.json) | spiral | 127.0053 | 408.406 | 0.02875 | 0.431 | 0.799 | 33.00 |
| [Z 193-14](../data/galaxy_sources/z_193_14.json) | spiral | 127.0053 | 408.406 | 0.02875 | 0.504 | 0.793 | 136.00 |
| [Z 193-12](../data/galaxy_sources/z_193_12.json) | spiral | 127.0053 | 408.406 | 0.02875 | 0.638 | 0.639 | 160.00 |
| [Z 38-9](../data/galaxy_sources/z_38_9.json) | elliptical | 127.0053 | 408.406 | 0.02875 | 0.459 | 0.696 | 39.00 |
| [Z 245-24](../data/galaxy_sources/z_245_24.json) | spiral | 127.0930 | 408.684 | 0.02877 | 0.720 | 0.612 | 100.00 |
| [MCG+06-32-066](../data/galaxy_sources/mcg_06_32_066.json) | barred spiral | 127.1369 | 408.823 | 0.02878 | 0.570 | 0.825 | 111.00 |
| [LEDA 1177626](../data/galaxy_sources/leda_1177626.json) | spiral | 127.1369 | 408.823 | 0.02878 | 0.213 | 0.920 | 105.00 |
| [LEDA 142890](../data/galaxy_sources/leda_142890.json) | spiral | 127.1807 | 408.962 | 0.02879 | 0.511 | 0.589 | 6.00 |
| [Z 193-13](../data/galaxy_sources/z_193_13.json) | elliptical | 127.2246 | 409.102 | 0.0288 | 0.613 | 0.971 | 4.00 |
| [LEDA 1148279](../data/galaxy_sources/leda_1148279.json) | spiral | 127.2246 | 409.102 | 0.0288 | 0.590 | 0.522 | 92.00 |
| [LEDA 1159337](../data/galaxy_sources/leda_1159337.json) | elliptical | 127.2246 | 409.102 | 0.0288 | 0.344 | 0.903 | 82.00 |
| [MCG+09-15-112](../data/galaxy_sources/mcg_09_15_112.json) | spiral | 127.2685 | 409.241 | 0.02881 | 0.454 | 0.888 | 120.00 |
| [MCG-04-12-008](../data/galaxy_sources/mcg_04_12_008.json) | spiral | 127.3123 | 409.380 | 0.02882 | 0.563 | 0.768 | 174.00 |
| [Z 428-10](../data/galaxy_sources/z_428_10.json) | spiral | 127.3123 | 409.380 | 0.02882 | 0.774 | 0.934 | 44.00 |
| [Z 245-40](../data/galaxy_sources/z_245_40.json) | spiral | 127.3562 | 409.519 | 0.02883 | 0.815 | 0.723 | 114.00 |
| [IC 528C](../data/galaxy_sources/ic_528c.json) | spiral | 127.3562 | 409.519 | 0.02883 | 0.313 | 0.629 | 105.00 |
| [LEDA 79472](../data/galaxy_sources/leda_79472.json) | spiral | 127.4001 | 409.658 | 0.02884 | 0.572 | 0.757 | 13.00 |
| [Z 120-29](../data/galaxy_sources/z_120_29.json) | spiral | 127.4001 | 409.658 | 0.02884 | 0.566 | 0.802 | 137.00 |
| [Z 39-132](../data/galaxy_sources/z_39_132.json) | elliptical | 127.4001 | 409.658 | 0.02884 | 0.933 | 0.934 | 52.00 |
| [Z 380-18](../data/galaxy_sources/z_380_18.json) | spiral | 127.4439 | 409.797 | 0.02885 | 0.793 | 0.774 | 176.00 |
| [Z 264-82](../data/galaxy_sources/z_264_82.json) | lenticular | 127.4878 | 409.936 | 0.02886 | 0.652 | 0.738 | 25.00 |
| [Z 222-23](../data/galaxy_sources/z_222_23.json) | spiral | 127.4878 | 409.936 | 0.02886 | 0.864 | 0.302 | 157.00 |
| [LEDA 32027](../data/galaxy_sources/leda_32027.json) | spiral | 127.4878 | 409.936 | 0.02886 | 0.406 | 0.902 | 48.00 |
| [Z 89-24](../data/galaxy_sources/z_89_24.json) | spiral | 127.4878 | 409.936 | 0.02886 | 0.237 | 0.937 | 120.00 |
| [LEDA 86922](../data/galaxy_sources/leda_86922.json) | spiral | 127.4878 | 409.936 | 0.02886 | 0.947 | 0.185 | 100.00 |
| [Z 208-50](../data/galaxy_sources/z_208_50.json) | spiral | 127.5317 | 410.075 | 0.02887 | 1.162 | 0.305 | 14.00 |
| [MCG+08-24-068](../data/galaxy_sources/mcg_08_24_068.json) | spiral | 127.5317 | 410.075 | 0.02887 | 0.395 | 0.908 | 102.00 |
| [LEDA 91109](../data/galaxy_sources/leda_91109.json) | spiral | 127.5317 | 410.075 | 0.02887 | 1.025 | 0.168 | 141.00 |
| [Z 237-16](../data/galaxy_sources/z_237_16.json) | barred spiral | 127.5756 | 410.214 | 0.02888 | 1.111 | 0.528 | 124.00 |
| [Z 427-6](../data/galaxy_sources/z_427_6.json) | weakly barred spiral | 127.5756 | 410.214 | 0.02888 | 1.211 | 0.202 | 172.00 |
| [Z 40-6](../data/galaxy_sources/z_40_6.json) | spiral | 127.5756 | 410.214 | 0.02888 | 0.375 | 0.760 | 72.00 |
| [Z 76-97](../data/galaxy_sources/z_76_97.json) | barred spiral | 127.5756 | 410.214 | 0.02888 | 0.819 | 0.741 | 9.00 |
| [Z 91-13](../data/galaxy_sources/z_91_13.json) | lenticular | 127.6194 | 410.353 | 0.02889 | 0.629 | 0.384 | 100.00 |
| [LEDA 82433](../data/galaxy_sources/leda_82433.json) | spiral | 127.6194 | 410.353 | 0.02889 | 0.568 | 0.421 | 48.00 |
| [LEDA 27801](../data/galaxy_sources/leda_27801.json) | spiral | 127.6194 | 410.353 | 0.02889 | 0.477 | 0.338 | 10.00 |
| [MCG+03-26-046](../data/galaxy_sources/mcg_03_26_046.json) | spiral | 127.6194 | 410.353 | 0.02889 | 0.520 | 0.928 | 112.00 |
| [Z 155-56](../data/galaxy_sources/z_155_56.json) | barred spiral | 127.6194 | 410.353 | 0.02889 | 0.712 | 0.865 | 177.00 |
| [Z 403-31](../data/galaxy_sources/z_403_31.json) | spiral | 127.6353 | 410.403 | 0.02889362 | 0.810 | 0.604 | 178.00 |
| [Z 45-106](../data/galaxy_sources/z_45_106.json) | elliptical | 127.6633 | 410.492 | 0.0289 | 0.636 | 0.787 | 57.00 |
| [Z 264-86](../data/galaxy_sources/z_264_86.json) | spiral | 127.7072 | 410.631 | 0.02891 | 0.440 | 0.880 | 75.00 |
| [Z 208-49](../data/galaxy_sources/z_208_49.json) | spiral | 127.7072 | 410.631 | 0.02891 | 0.545 | 0.501 | 40.00 |
| [Z 5-13](../data/galaxy_sources/z_5_13.json) | spiral | 127.7072 | 410.631 | 0.02891 | 0.804 | 0.526 | 147.00 |
| [Z 62-31](../data/galaxy_sources/z_62_31.json) | spiral | 127.7072 | 410.631 | 0.02891 | 1.123 | 0.184 | 109.00 |
| [IC 2619](../data/galaxy_sources/ic_2619.json) | lenticular | 127.7072 | 410.631 | 0.02891 | 0.447 | 0.714 | 68.00 |
| [Z 300-48](../data/galaxy_sources/z_300_48.json) | lenticular | 127.7510 | 410.770 | 0.02892 | 0.263 | 0.882 | 165.00 |
| [LEDA 69642](../data/galaxy_sources/leda_69642.json) | spiral | 127.7510 | 410.770 | 0.02892 | 0.637 | 0.380 | 175.00 |
| [Z 11-38](../data/galaxy_sources/z_11_38.json) | spiral | 127.7510 | 410.770 | 0.02892 | 0.798 | 0.513 | 121.00 |
| [Z 101-28](../data/galaxy_sources/z_101_28.json) | weakly barred spiral | 127.7709 | 410.833 | 0.02892453 | 1.248 | 0.541 | 175.00 |
| [LEDA 3098317](../data/galaxy_sources/leda_3098317.json) | lenticular | 127.7949 | 410.909 | 0.02893 | 0.625 | 0.980 | 164.00 |
| [Z 89-28](../data/galaxy_sources/z_89_28.json) | spiral | 127.8388 | 411.048 | 0.02894 | 0.905 | 0.252 | 9.00 |
| [LEDA 2816028](../data/galaxy_sources/leda_2816028.json) | lenticular | 127.9265 | 411.327 | 0.02896 | 0.347 | 0.842 | 2.00 |
| [MCG+08-24-007](../data/galaxy_sources/mcg_08_24_007.json) | spiral | 127.9704 | 411.466 | 0.02897 | 0.564 | 0.539 | 78.00 |
| [Z 11-112](../data/galaxy_sources/z_11_112.json) | elliptical | 128.0142 | 411.605 | 0.02898 | 0.668 | 0.713 | 148.00 |
| [MCG+09-22-070](../data/galaxy_sources/mcg_09_22_070.json) | spiral | 128.0581 | 411.744 | 0.02899 | 0.576 | 0.679 | 129.00 |
| [MCG+08-24-069](../data/galaxy_sources/mcg_08_24_069.json) | weakly barred spiral | 128.0581 | 411.744 | 0.02899 | 0.969 | 0.599 | 32.00 |
| [LEDA 90854](../data/galaxy_sources/leda_90854.json) | spiral | 128.0581 | 411.744 | 0.02899 | 0.739 | 0.162 | 149.00 |
| [NGC 6330](../data/galaxy_sources/ngc_6330.json) | spiral | 128.0581 | 411.744 | 0.02899 | 1.206 | 0.390 | 166.00 |
| [Z 157-78](../data/galaxy_sources/z_157_78.json) | barred spiral | 128.1019 | 411.883 | 0.029 | 0.485 | 0.696 | 35.00 |
| [LEDA 2582096](../data/galaxy_sources/leda_2582096.json) | elliptical | 128.1458 | 412.022 | 0.02901 | 0.498 | 0.782 | 31.00 |
| [MCG+09-22-001](../data/galaxy_sources/mcg_09_22_001.json) | spiral | 128.1458 | 412.022 | 0.02901 | 0.290 | 0.359 | 60.00 |
| [Z 193-3](../data/galaxy_sources/z_193_3.json) | lenticular | 128.1458 | 412.022 | 0.02901 | 0.603 | 0.764 | 34.00 |
| [MCG-04-12-001](../data/galaxy_sources/mcg_04_12_001.json) | spiral | 128.1897 | 412.161 | 0.02902 | 1.282 | 0.306 | 122.00 |
| [LEDA 173088](../data/galaxy_sources/leda_173088.json) | elliptical | 128.1897 | 412.161 | 0.02902 | 0.415 | 0.745 | 27.00 |
| [Z 16-20](../data/galaxy_sources/z_16_20.json) | spiral | 128.1897 | 412.161 | 0.02902 | 0.745 | 0.499 | 168.00 |
| [Z 415-59](../data/galaxy_sources/z_415_59.json) | spiral | 128.2682 | 412.409 | 0.02903789 | 0.480 | 0.760 | 20.00 |
| [LEDA 2433109](../data/galaxy_sources/leda_2433109.json) | spiral | 128.2774 | 412.439 | 0.02904 | 0.486 | 0.595 | 108.00 |
| [LEDA 94189](../data/galaxy_sources/leda_94189.json) | elliptical | 128.2774 | 412.439 | 0.02904 | 0.627 | 0.897 | 135.00 |
| [Z 374-23](../data/galaxy_sources/z_374_23.json) | spiral | 128.2774 | 412.439 | 0.02904 | 1.398 | 0.377 | 169.00 |
| [MCG+06-33-006](../data/galaxy_sources/mcg_06_33_006.json) | lenticular | 128.3213 | 412.578 | 0.02905 | 0.653 | 0.632 | 116.00 |
| [Z 36-25](../data/galaxy_sources/z_36_25.json) | spiral | 128.3213 | 412.578 | 0.02905 | 0.552 | 0.588 | 101.00 |
| [LEDA 91076](../data/galaxy_sources/leda_91076.json) | spiral | 128.3651 | 412.717 | 0.02906 | 1.035 | 0.136 | 100.00 |
| [Z 12-17](../data/galaxy_sources/z_12_17.json) | spiral | 128.3651 | 412.717 | 0.02906 | 0.413 | 0.793 | 142.00 |
| [Z 120-27](../data/galaxy_sources/z_120_27.json) | lenticular | 128.4090 | 412.856 | 0.02907 | 0.559 | 0.449 | 135.00 |
| [Z 412-16](../data/galaxy_sources/z_412_16.json) | spiral | 128.4339 | 412.935 | 0.02907568 | 0.964 | 0.245 | 95.00 |
| [Z 208-46](../data/galaxy_sources/z_208_46.json) | elliptical | 128.4529 | 412.995 | 0.02908 | 0.991 | 0.572 | 60.00 |
| [Z 208-43](../data/galaxy_sources/z_208_43.json) | spiral | 128.4529 | 412.995 | 0.02908 | 0.865 | 0.377 | 153.00 |
| [Z 408-14](../data/galaxy_sources/z_408_14.json) | spiral | 128.5243 | 413.221 | 0.02909629 | 0.809 | 0.616 | 158.00 |
| [Z 62-18](../data/galaxy_sources/z_62_18.json) | spiral | 128.5406 | 413.273 | 0.0291 | 0.790 | 0.651 | 104.00 |
| [Z 39-79](../data/galaxy_sources/z_39_79.json) | lenticular | 128.5406 | 413.273 | 0.0291 | 0.509 | 0.580 | 77.00 |
| [Z 211-18](../data/galaxy_sources/z_211_18.json) | spiral | 128.5844 | 413.412 | 0.02911 | 0.757 | 0.294 | 169.00 |
| [Z 245-23](../data/galaxy_sources/z_245_23.json) | spiral | 128.6283 | 413.551 | 0.02912 | 0.725 | 0.568 | 176.00 |
| [Z 212-21](../data/galaxy_sources/z_212_21.json) | spiral | 128.6722 | 413.690 | 0.02913 | 0.567 | 0.525 | 47.00 |
| [Z 152-73](../data/galaxy_sources/z_152_73.json) | spiral | 128.6722 | 413.690 | 0.02913 | 0.981 | 0.673 | 122.00 |
| [Z 47-87](../data/galaxy_sources/z_47_87.json) | lenticular | 128.6722 | 413.690 | 0.02913 | 0.919 | 0.372 | 145.00 |
| [Z 208-28](../data/galaxy_sources/z_208_28.json) | spiral | 128.7599 | 413.968 | 0.02915 | 0.757 | 0.662 | 163.00 |
| [LEDA 87001](../data/galaxy_sources/leda_87001.json) | spiral | 128.8038 | 414.107 | 0.02916 | 0.428 | 0.463 | 153.00 |
| [Z 192-42](../data/galaxy_sources/z_192_42.json) | elliptical | 128.8038 | 414.107 | 0.02916 | 0.814 | 0.976 | 33.00 |
| [Z 31-54](../data/galaxy_sources/z_31_54.json) | spiral | 128.8476 | 414.246 | 0.02917 | 0.741 | 0.945 | 2.00 |
| [Z 36-28](../data/galaxy_sources/z_36_28.json) | spiral | 128.8476 | 414.246 | 0.02917 | 0.894 | 0.391 | 76.00 |
| [Z 5-7](../data/galaxy_sources/z_5_7.json) | spiral | 128.8915 | 414.385 | 0.02918 | 0.699 | 0.535 | 72.00 |
| [LEDA 1156834](../data/galaxy_sources/leda_1156834.json) | elliptical | 128.8915 | 414.385 | 0.02918 | 0.430 | 0.822 | 142.00 |
| [Z 137-47](../data/galaxy_sources/z_137_47.json) | spiral | 128.8915 | 414.385 | 0.02918 | 0.585 | 0.913 | 148.00 |
| [Z 208-27](../data/galaxy_sources/z_208_27.json) | spiral | 128.9353 | 414.524 | 0.02919 | 0.453 | 0.953 | 7.00 |
| [Z 39-176](../data/galaxy_sources/z_39_176.json) | spiral | 128.9353 | 414.524 | 0.02919 | 0.579 | 0.714 | 166.00 |
| [Z 404-1](../data/galaxy_sources/z_404_1.json) | barred spiral | 128.9764 | 414.654 | 0.02919936 | 1.037 | 0.378 | 36.00 |
| [MCG+03-24-049](../data/galaxy_sources/mcg_03_24_049.json) | spiral | 128.9792 | 414.663 | 0.0292 | 0.699 | 0.622 | 123.00 |
| [Z 37-84](../data/galaxy_sources/z_37_84.json) | lenticular | 128.9792 | 414.663 | 0.0292 | 0.633 | 0.613 | 78.00 |
| [Z 238-35](../data/galaxy_sources/z_238_35.json) | spiral | 129.0231 | 414.802 | 0.02921 | 0.822 | 0.628 | 120.00 |
| [MCG+06-32-098](../data/galaxy_sources/mcg_06_32_098.json) | elliptical | 129.0231 | 414.802 | 0.02921 | 0.619 | 0.782 | 118.00 |
| [Z 287-72](../data/galaxy_sources/z_287_72.json) | elliptical | 129.0669 | 414.941 | 0.02922 | 0.617 | 0.859 | 95.00 |
| [NGC 5862](../data/galaxy_sources/ngc_5862.json) | spiral | 129.0669 | 414.941 | 0.02922 | 0.738 | 0.828 | 11.00 |
| [Z 9-86](../data/galaxy_sources/z_9_86.json) | spiral | 129.0669 | 414.941 | 0.02922 | 1.014 | 0.644 | 86.00 |
| [Z 211-40](../data/galaxy_sources/z_211_40.json) | barred spiral | 129.1108 | 415.080 | 0.02923 | 0.440 | 0.955 | 51.00 |
| [Z 18-54](../data/galaxy_sources/z_18_54.json) | spiral | 129.1108 | 415.080 | 0.02923 | 0.677 | 0.329 | 128.00 |
| [LEDA 159923](../data/galaxy_sources/leda_159923.json) | barred lenticular | 129.1108 | 415.080 | 0.02923 | 0.450 | 0.890 | 134.00 |
| [MCG+10-25-057](../data/galaxy_sources/mcg_10_25_057.json) | spiral | 129.1546 | 415.219 | 0.02924 | 0.429 | 0.879 | 170.00 |
| [Z 39-84](../data/galaxy_sources/z_39_84.json) | spiral | 129.1546 | 415.219 | 0.02924 | 0.791 | 0.287 | 105.00 |
| [Z 14-41](../data/galaxy_sources/z_14_41.json) | spiral | 129.1546 | 415.219 | 0.02924 | 0.712 | 0.928 | 47.00 |
| [MCG+08-20-009](../data/galaxy_sources/mcg_08_20_009.json) | barred spiral | 129.2862 | 415.636 | 0.02927 | 0.691 | 0.555 | 112.00 |
| [Z 389-36](../data/galaxy_sources/z_389_36.json) | spiral | 129.2862 | 415.636 | 0.02927 | 0.930 | 0.444 | 98.00 |
| [LEDA 86929](../data/galaxy_sources/leda_86929.json) | spiral | 129.2862 | 415.636 | 0.02927 | 0.467 | 0.417 | 21.00 |
| [LEDA 33920](../data/galaxy_sources/leda_33920.json) | spiral | 129.2862 | 415.636 | 0.02927 | 0.571 | 0.696 | 64.00 |
| [LEDA 105549](../data/galaxy_sources/leda_105549.json) | barred spiral | 129.2862 | 415.636 | 0.02927 | 0.410 | 0.701 | 2.00 |
| [Z 88-29](../data/galaxy_sources/z_88_29.json) | spiral | 129.3301 | 415.775 | 0.02928 | 0.580 | 0.727 | 114.00 |
| [Z 36-19](../data/galaxy_sources/z_36_19.json) | spiral | 129.3301 | 415.775 | 0.02928 | 1.084 | 0.373 | 159.00 |
| [ESO 99-4](../data/galaxy_sources/eso_99_4.json) | spiral | 129.3345 | 415.789 | 0.029281 | 0.607 | 0.460 | 30.00 |
| [LEDA 28566](../data/galaxy_sources/leda_28566.json) | spiral | 129.3739 | 415.914 | 0.02929 | 0.382 | 0.538 | 61.00 |
| [Z 47-103](../data/galaxy_sources/z_47_103.json) | barred lenticular | 129.3739 | 415.914 | 0.02929 | 0.430 | 0.920 | 66.00 |
| [MCG+10-24-113](../data/galaxy_sources/mcg_10_24_113.json) | spiral | 129.4178 | 416.053 | 0.0293 | 0.538 | 0.736 | 70.00 |
| [Z 192-27](../data/galaxy_sources/z_192_27.json) | barred spiral | 129.4178 | 416.053 | 0.0293 | 0.892 | 0.700 | 57.00 |
| [MCG+09-18-058](../data/galaxy_sources/mcg_09_18_058.json) | spiral | 129.5055 | 416.331 | 0.02932 | 0.426 | 0.710 | 150.00 |
| [IC 2521](../data/galaxy_sources/ic_2521.json) | spiral | 129.6371 | 416.748 | 0.02935 | 0.580 | 0.875 | 5.00 |
| [UGC 9806](../data/galaxy_sources/ugc_9806.json) | spiral | 129.6809 | 416.887 | 0.02936 | 0.933 | 0.307 | 102.00 |
| [MCG+07-23-036](../data/galaxy_sources/mcg_07_23_036.json) | spiral | 129.6809 | 416.887 | 0.02936 | 0.537 | 0.723 | 67.00 |
| [MCG+08-28-039](../data/galaxy_sources/mcg_08_28_039.json) | spiral | 129.7248 | 417.026 | 0.02937 | 0.875 | 0.323 | 144.00 |
| [MCG+08-24-009](../data/galaxy_sources/mcg_08_24_009.json) | spiral | 129.7248 | 417.026 | 0.02937 | 0.646 | 0.592 | 167.00 |
| [LEDA 145184](../data/galaxy_sources/leda_145184.json) | lenticular | 129.7248 | 417.026 | 0.02937 | 0.649 | 0.351 | 100.00 |
| [NGC 2895](../data/galaxy_sources/ngc_2895.json) | barred spiral | 129.7687 | 417.165 | 0.02938 | 0.831 | 0.923 | 48.00 |
| [MCG+09-21-072](../data/galaxy_sources/mcg_09_21_072.json) | spiral | 129.7687 | 417.165 | 0.02938 | 0.627 | 0.833 | 40.00 |
| [MCG+08-23-012](../data/galaxy_sources/mcg_08_23_012.json) | spiral | 129.7687 | 417.165 | 0.02938 | 0.942 | 0.401 | 121.00 |
| [LEDA 78754](../data/galaxy_sources/leda_78754.json) | spiral | 129.7687 | 417.165 | 0.02938 | 0.374 | 0.594 | 109.00 |
| [Z 426-12](../data/galaxy_sources/z_426_12.json) | spiral | 129.7687 | 417.165 | 0.02938 | 0.423 | 0.847 | 174.00 |
| [Z 390-87](../data/galaxy_sources/z_390_87.json) | barred spiral | 129.8125 | 417.304 | 0.02939 | 0.647 | 0.496 | 85.00 |
| [LEDA 70217](../data/galaxy_sources/leda_70217.json) | spiral | 129.8125 | 417.304 | 0.02939 | 0.441 | 0.496 | 20.00 |
| [Z 192-28](../data/galaxy_sources/z_192_28.json) | spiral | 129.8125 | 417.304 | 0.02939 | 0.649 | 0.577 | 120.00 |
| [NGC 5503](../data/galaxy_sources/ngc_5503.json) | spiral | 129.8564 | 417.443 | 0.0294 | 0.670 | 0.476 | 81.00 |
| [MCG-01-54-019](../data/galaxy_sources/mcg_01_54_019.json) | spiral | 129.8564 | 417.443 | 0.0294 | 1.591 | 0.396 | 175.00 |
| [ESO 88-20](../data/galaxy_sources/eso_88_20.json) | spiral | 129.8656 | 417.472 | 0.0294021 | 0.323 | 0.681 | 175.00 |
| [Z 275-28](../data/galaxy_sources/z_275_28.json) | elliptical | 129.9441 | 417.720 | 0.02942 | 0.539 | 0.774 | 169.00 |
| [MCG+09-20-108](../data/galaxy_sources/mcg_09_20_108.json) | spiral | 129.9441 | 417.720 | 0.02942 | 0.924 | 0.397 | 113.00 |
| [Z 240-35](../data/galaxy_sources/z_240_35.json) | barred spiral | 129.9879 | 417.859 | 0.02943 | 0.472 | 0.759 | 38.00 |
| [UGC 5039](../data/galaxy_sources/ugc_5039.json) | spiral | 129.9879 | 417.859 | 0.02943 | 1.042 | 0.290 | 145.00 |
| [IC 4506](../data/galaxy_sources/ic_4506.json) | spiral | 129.9879 | 417.859 | 0.02943 | 0.557 | 0.899 | 179.00 |
| [Z 44-74](../data/galaxy_sources/z_44_74.json) | spiral | 130.0318 | 417.998 | 0.02944 | 0.709 | 0.370 | 21.00 |
| [IC 4459](../data/galaxy_sources/ic_4459.json) | spiral | 130.0757 | 418.137 | 0.02945 | 0.790 | 0.322 | 11.00 |
| [Z 264-68](../data/galaxy_sources/z_264_68.json) | spiral | 130.1195 | 418.276 | 0.02946 | 0.451 | 0.675 | 8.00 |
| [MCG+08-24-059](../data/galaxy_sources/mcg_08_24_059.json) | barred spiral | 130.1195 | 418.276 | 0.02946 | 0.591 | 0.629 | 85.00 |
| [Z 208-24](../data/galaxy_sources/z_208_24.json) | spiral | 130.1634 | 418.415 | 0.02947 | 1.230 | 0.194 | 12.00 |
| [NGC 2739](../data/galaxy_sources/ngc_2739.json) | spiral | 130.2072 | 418.554 | 0.02948 | 1.048 | 0.372 | 96.00 |
| [LEDA 1126829](../data/galaxy_sources/leda_1126829.json) | spiral | 130.2072 | 418.554 | 0.02948 | 0.381 | 0.460 | 27.00 |
| [Z 63-79](../data/galaxy_sources/z_63_79.json) | spiral | 130.2072 | 418.554 | 0.02948 | 0.805 | 0.360 | 6.00 |
| [Z 273-37](../data/galaxy_sources/z_273_37.json) | elliptical | 130.2511 | 418.693 | 0.02949 | 0.748 | 0.952 | 7.00 |
| [MCG+09-21-085](../data/galaxy_sources/mcg_09_21_085.json) | spiral | 130.2511 | 418.693 | 0.02949 | 0.557 | 0.315 | 166.00 |
| [Z 118-51](../data/galaxy_sources/z_118_51.json) | spiral | 130.2511 | 418.693 | 0.02949 | 0.557 | 0.461 | 144.00 |
| [Z 49-44](../data/galaxy_sources/z_49_44.json) | elliptical | 130.2511 | 418.693 | 0.02949 | 1.052 | 0.890 | 158.00 |
| [Z 221-30](../data/galaxy_sources/z_221_30.json) | spiral | 130.2949 | 418.832 | 0.0295 | 0.715 | 0.761 | 20.00 |
| [Z 289-13](../data/galaxy_sources/z_289_13.json) | spiral | 130.3388 | 418.971 | 0.02951 | 0.563 | 0.676 | 150.00 |
| [NGC 2740](../data/galaxy_sources/ngc_2740.json) | barred spiral | 130.3388 | 418.971 | 0.02951 | 0.721 | 0.871 | 57.00 |
| [LEDA 1153797](../data/galaxy_sources/leda_1153797.json) | spiral | 130.3388 | 418.971 | 0.02951 | 0.734 | 0.213 | 151.00 |
| [Z 291-15](../data/galaxy_sources/z_291_15.json) | lenticular | 130.3826 | 419.110 | 0.02952 | 0.544 | 0.939 | 69.00 |
| [LEDA 70853](../data/galaxy_sources/leda_70853.json) | spiral | 130.4082 | 419.191 | 0.02952583 | 0.637 | 0.382 | 97.00 |
| [Z 277-43](../data/galaxy_sources/z_277_43.json) | elliptical | 130.4265 | 419.249 | 0.02953 | 0.535 | 0.870 | 37.00 |
| [LEDA 54042](../data/galaxy_sources/leda_54042.json) | spiral | 130.4703 | 419.388 | 0.02954 | 0.470 | 0.618 | 123.00 |
| [MCG+09-24-046](../data/galaxy_sources/mcg_09_24_046.json) | lenticular | 130.4703 | 419.388 | 0.02954 | 0.167 | 0.898 | 70.00 |
| [Z 94-91](../data/galaxy_sources/z_94_91.json) | spiral | 130.4703 | 419.388 | 0.02954 | 0.478 | 0.898 | 65.00 |
| [LEDA 1133128](../data/galaxy_sources/leda_1133128.json) | spiral | 130.4703 | 419.388 | 0.02954 | 0.528 | 0.324 | 171.00 |
| [Z 19-65](../data/galaxy_sources/z_19_65.json) | barred spiral | 130.6019 | 419.804 | 0.02957 | 0.402 | 0.667 | 62.00 |
| [MCG+10-22-011](../data/galaxy_sources/mcg_10_22_011.json) | barred spiral | 130.6458 | 419.943 | 0.02958 | 0.807 | 0.456 | 119.00 |
| [Z 18-10](../data/galaxy_sources/z_18_10.json) | spiral | 130.6458 | 419.943 | 0.02958 | 0.710 | 0.929 | 112.00 |
| [Z 170-3](../data/galaxy_sources/z_170_3.json) | spiral | 130.6896 | 420.082 | 0.02959 | 0.823 | 0.552 | 160.00 |
| [LEDA 1831130](../data/galaxy_sources/leda_1831130.json) | spiral | 130.7277 | 420.203 | 0.0295987 | 0.471 | 0.541 | 76.00 |
| [Z 58-65](../data/galaxy_sources/z_58_65.json) | spiral | 130.7773 | 420.360 | 0.02961 | 0.776 | 0.388 | 56.00 |
| [Z 67-15](../data/galaxy_sources/z_67_15.json) | spiral | 130.7773 | 420.360 | 0.02961 | 0.641 | 0.472 | 45.00 |
| [Z 158-26](../data/galaxy_sources/z_158_26.json) | lenticular | 130.7773 | 420.360 | 0.02961 | 0.916 | 0.857 | 173.00 |
| [LEDA 1026745](../data/galaxy_sources/leda_1026745.json) | spiral | 130.7773 | 420.360 | 0.02961 | 0.626 | 0.745 | 118.00 |
| [Z 269-50](../data/galaxy_sources/z_269_50.json) | elliptical | 130.8212 | 420.499 | 0.02962 | 0.596 | 0.953 | 94.00 |
| [Z 31-76](../data/galaxy_sources/z_31_76.json) | barred spiral | 130.8212 | 420.499 | 0.02962 | 0.787 | 0.521 | 39.00 |
| [Z 91-14](../data/galaxy_sources/z_91_14.json) | spiral | 130.8212 | 420.499 | 0.02962 | 0.702 | 0.443 | 142.00 |
| [NGC 3019](../data/galaxy_sources/ngc_3019.json) | spiral | 130.8212 | 420.499 | 0.02962 | 0.717 | 0.595 | 37.00 |
| [LEDA 1098475](../data/galaxy_sources/leda_1098475.json) | barred spiral | 130.8212 | 420.499 | 0.02962 | 0.334 | 0.836 | 7.00 |
| [ESO 274-12](../data/galaxy_sources/eso_274_12.json) | spiral | 130.8574 | 420.614 | 0.02962826 | 0.413 | 0.840 | 10.00 |
| [Z 61-50](../data/galaxy_sources/z_61_50.json) | spiral | 130.8650 | 420.638 | 0.02963 | 0.597 | 0.932 | 5.00 |
| [MCG+06-32-064](../data/galaxy_sources/mcg_06_32_064.json) | spiral | 130.8650 | 420.638 | 0.02963 | 0.851 | 0.413 | 163.00 |
| [NGC 7674A](../data/galaxy_sources/ngc_7674a.json) | spiral | 130.8905 | 420.719 | 0.02963582 | 0.347 | 0.680 | 30.00 |
| [Z 269-53](../data/galaxy_sources/z_269_53.json) | lenticular | 130.9089 | 420.777 | 0.02964 | 0.705 | 0.605 | 159.00 |
| [Z 45-119](../data/galaxy_sources/z_45_119.json) | spiral | 130.9089 | 420.777 | 0.02964 | 0.617 | 0.733 | 162.00 |
| [Z 32-23](../data/galaxy_sources/z_32_23.json) | spiral | 130.9966 | 421.055 | 0.02966 | 0.765 | 0.436 | 77.00 |
| [Z 275-10](../data/galaxy_sources/z_275_10.json) | spiral | 131.0404 | 421.194 | 0.02967 | 0.696 | 0.697 | 77.00 |
| [LEDA 1160266](../data/galaxy_sources/leda_1160266.json) | elliptical | 131.0404 | 421.194 | 0.02967 | 0.342 | 0.796 | 3.00 |
| [IC 2249](../data/galaxy_sources/ic_2249.json) | spiral | 131.0843 | 421.332 | 0.02968 | 0.360 | 0.773 | 127.00 |
| [LEDA 91407](../data/galaxy_sources/leda_91407.json) | spiral | 131.1281 | 421.471 | 0.02969 | 0.775 | 0.195 | 133.00 |
| [Z 69-51](../data/galaxy_sources/z_69_51.json) | spiral | 131.1281 | 421.471 | 0.02969 | 0.404 | 0.872 | 178.00 |
| [UGC 10857](../data/galaxy_sources/ugc_10857.json) | lenticular | 131.1720 | 421.610 | 0.0297 | 0.494 | 0.547 | 128.00 |
| [Z 194-1](../data/galaxy_sources/z_194_1.json) | spiral | 131.1720 | 421.610 | 0.0297 | 0.587 | 0.796 | 171.00 |
| [Z 60-3](../data/galaxy_sources/z_60_3.json) | spiral | 131.1720 | 421.610 | 0.0297 | 0.800 | 0.640 | 42.00 |
| [Z 155-57](../data/galaxy_sources/z_155_57.json) | spiral | 131.1720 | 421.610 | 0.0297 | 1.161 | 0.556 | 123.00 |
| [Z 11-78](../data/galaxy_sources/z_11_78.json) | spiral | 131.1720 | 421.610 | 0.0297 | 0.601 | 0.767 | 156.00 |
| [Z 271-23](../data/galaxy_sources/z_271_23.json) | spiral | 131.2597 | 421.888 | 0.02972 | 1.449 | 0.349 | 18.00 |
| [LEDA 12492](../data/galaxy_sources/leda_12492.json) | spiral | 131.2825 | 421.960 | 0.0297252 | 0.290 | 0.632 | 29.00 |
| [MCG+09-21-059](../data/galaxy_sources/mcg_09_21_059.json) | lenticular | 131.3912 | 422.305 | 0.02975 | 0.651 | 0.389 | 63.00 |
| [Z 48-89](../data/galaxy_sources/z_48_89.json) | spiral | 131.3912 | 422.305 | 0.02975 | 0.764 | 0.939 | 164.00 |
| [MCG+08-30-029](../data/galaxy_sources/mcg_08_30_029.json) | spiral | 131.4351 | 422.444 | 0.02976 | 0.660 | 0.714 | 34.00 |
| [Z 193-17](../data/galaxy_sources/z_193_17.json) | elliptical | 131.4351 | 422.444 | 0.02976 | 0.860 | 0.631 | 122.00 |
| [Z 67-18](../data/galaxy_sources/z_67_18.json) | lenticular | 131.4351 | 422.444 | 0.02976 | 0.847 | 0.797 | 54.00 |
| [Z 93-77](../data/galaxy_sources/z_93_77.json) | elliptical | 131.4789 | 422.582 | 0.02977 | 0.407 | 0.958 | 145.00 |
| [ESO 63-2](../data/galaxy_sources/eso_63_2.json) | spiral | 131.5146 | 422.695 | 0.02977814 | 0.347 | 0.660 | 105.00 |
| [IC 4460](../data/galaxy_sources/ic_4460.json) | spiral | 131.5228 | 422.721 | 0.02978 | 0.472 | 0.819 | 164.00 |
| [Z 19-58](../data/galaxy_sources/z_19_58.json) | spiral | 131.5228 | 422.721 | 0.02978 | 0.440 | 0.622 | 117.00 |
| [Z 314-41](../data/galaxy_sources/z_314_41.json) | spiral | 131.5666 | 422.860 | 0.02979 | 0.723 | 0.501 | 75.00 |
| [LEDA 33491](../data/galaxy_sources/leda_33491.json) | spiral | 131.5666 | 422.860 | 0.02979 | 0.540 | 0.962 | 67.00 |
| [Z 294-48](../data/galaxy_sources/z_294_48.json) | spiral | 131.6105 | 422.999 | 0.0298 | 0.740 | 0.890 | 125.00 |
| [Z 271-12](../data/galaxy_sources/z_271_12.json) | spiral | 131.6105 | 422.999 | 0.0298 | 0.549 | 0.660 | 79.00 |
| [LEDA 84523](../data/galaxy_sources/leda_84523.json) | barred spiral | 131.6105 | 422.999 | 0.0298 | 0.743 | 0.429 | 69.00 |
| [IC 578](../data/galaxy_sources/ic_578.json) | spiral | 131.6105 | 422.999 | 0.0298 | 1.142 | 0.417 | 68.00 |
| [Z 275-24](../data/galaxy_sources/z_275_24.json) | spiral | 131.6543 | 423.138 | 0.02981 | 0.662 | 0.507 | 135.00 |
| [MCG+09-27-005](../data/galaxy_sources/mcg_09_27_005.json) | spiral | 131.6982 | 423.277 | 0.02982 | 0.666 | 0.346 | 130.00 |
| [Z 290-53](../data/galaxy_sources/z_290_53.json) | lenticular | 131.6982 | 423.277 | 0.02982 | 0.536 | 0.937 | 29.00 |
| [Z 374-25](../data/galaxy_sources/z_374_25.json) | spiral | 131.6982 | 423.277 | 0.02982 | 0.941 | 0.661 | 103.00 |
| [MCG+09-15-076](../data/galaxy_sources/mcg_09_15_076.json) | spiral | 131.7420 | 423.416 | 0.02983 | 0.637 | 0.612 | 16.00 |
| [Z 37-66](../data/galaxy_sources/z_37_66.json) | spiral | 131.7420 | 423.416 | 0.02983 | 0.474 | 0.879 | 29.00 |
| [Z 32-42](../data/galaxy_sources/z_32_42.json) | spiral | 131.7799 | 423.536 | 0.02983865 | 0.735 | 0.316 | 35.00 |
| [LEDA 1023268](../data/galaxy_sources/leda_1023268.json) | elliptical | 131.7859 | 423.555 | 0.02984 | 0.493 | 0.924 | 48.00 |
| [Z 387-70](../data/galaxy_sources/z_387_70.json) | spiral | 131.7859 | 423.555 | 0.02984 | 0.603 | 0.449 | 96.00 |
| [LEDA 87025](../data/galaxy_sources/leda_87025.json) | spiral | 131.7859 | 423.555 | 0.02984 | 0.426 | 0.798 | 173.00 |
| [Z 18-1](../data/galaxy_sources/z_18_1.json) | spiral | 131.7859 | 423.555 | 0.02984 | 0.617 | 0.910 | 93.00 |
| [NGC 3016](../data/galaxy_sources/ngc_3016.json) | spiral | 131.8297 | 423.693 | 0.02985 | 1.498 | 0.633 | 73.00 |
| [Z 14-36](../data/galaxy_sources/z_14_36.json) | lenticular | 131.8297 | 423.693 | 0.02985 | 0.794 | 0.737 | 36.00 |
| [Z 376-1](../data/galaxy_sources/z_376_1.json) | spiral | 131.8735 | 423.832 | 0.02986 | 0.805 | 0.503 | 34.00 |
| [Z 14-4](../data/galaxy_sources/z_14_4.json) | elliptical | 131.8735 | 423.832 | 0.02986 | 0.267 | 0.599 | 125.00 |
| [LEDA 80265](../data/galaxy_sources/leda_80265.json) | spiral | 131.9174 | 423.971 | 0.02987 | 0.703 | 0.305 | 129.00 |
| [Z 277-38](../data/galaxy_sources/z_277_38.json) | spiral | 131.9612 | 424.110 | 0.02988 | 0.817 | 0.521 | 18.00 |
| [MCG+10-14-047](../data/galaxy_sources/mcg_10_14_047.json) | spiral | 131.9612 | 424.110 | 0.02988 | 0.716 | 0.574 | 146.00 |
| [Z 61-9](../data/galaxy_sources/z_61_9.json) | spiral | 131.9612 | 424.110 | 0.02988 | 0.652 | 0.421 | 173.00 |
| [Z 61-13](../data/galaxy_sources/z_61_13.json) | spiral | 132.0051 | 424.249 | 0.02989 | 0.551 | 0.917 | 176.00 |
| [LEDA 3093275](../data/galaxy_sources/leda_3093275.json) | elliptical | 132.0051 | 424.249 | 0.02989 | 0.645 | 0.632 | 174.00 |
| [Z 75-25](../data/galaxy_sources/z_75_25.json) | spiral | 132.0489 | 424.388 | 0.0299 | 0.661 | 0.586 | 165.00 |
| [MCG+08-29-050](../data/galaxy_sources/mcg_08_29_050.json) | spiral | 132.0928 | 424.527 | 0.02991 | 1.033 | 0.196 | 147.00 |
| [MCG+07-27-035](../data/galaxy_sources/mcg_07_27_035.json) | spiral | 132.0928 | 424.527 | 0.02991 | 0.859 | 0.309 | 85.00 |
| [UGC 6085](../data/galaxy_sources/ugc_6085.json) | spiral | 132.1366 | 424.665 | 0.02992 | 0.954 | 0.197 | 53.00 |
| [IC 4483](../data/galaxy_sources/ic_4483.json) | spiral | 132.1366 | 424.665 | 0.02992 | 1.538 | 0.346 | 26.00 |
| [LEDA 1170723](../data/galaxy_sources/leda_1170723.json) | elliptical | 132.1366 | 424.665 | 0.02992 | 0.420 | 0.659 | 170.00 |
| [Z 277-28](../data/galaxy_sources/z_277_28.json) | spiral | 132.1805 | 424.804 | 0.02993 | 0.385 | 0.784 | 17.00 |
| [Z 95-74](../data/galaxy_sources/z_95_74.json) | barred spiral | 132.1805 | 424.804 | 0.02993 | 0.675 | 0.911 | 24.00 |
| [Z 270-36](../data/galaxy_sources/z_270_36.json) | spiral | 132.2243 | 424.943 | 0.02994 | 0.634 | 0.833 | 116.00 |
| [LEDA 213497](../data/galaxy_sources/leda_213497.json) | spiral | 132.2243 | 424.943 | 0.02994 | 0.411 | 0.761 | 81.00 |
| [Z 91-17](../data/galaxy_sources/z_91_17.json) | spiral | 132.2243 | 424.943 | 0.02994 | 1.551 | 0.141 | 5.00 |
| [Z 271-22](../data/galaxy_sources/z_271_22.json) | spiral | 132.2682 | 425.082 | 0.02995 | 0.564 | 0.891 | 79.00 |
| [Z 238-48](../data/galaxy_sources/z_238_48.json) | spiral | 132.2682 | 425.082 | 0.02995 | 0.602 | 0.715 | 119.00 |
| [Z 149-33](../data/galaxy_sources/z_149_33.json) | spiral | 132.2682 | 425.082 | 0.02995 | 0.457 | 0.940 | 71.00 |
| [MCG-01-08-002](../data/galaxy_sources/mcg_01_08_002.json) | spiral | 132.2682 | 425.082 | 0.02995 | 1.096 | 0.808 | 72.00 |
| [Z 61-53](../data/galaxy_sources/z_61_53.json) | spiral | 132.3120 | 425.221 | 0.02996 | 0.830 | 0.341 | 148.00 |
| [MCG-01-54-012](../data/galaxy_sources/mcg_01_54_012.json) | spiral | 132.3528 | 425.350 | 0.02996931 | 0.978 | 0.533 | 51.00 |
| [LEDA 82678](../data/galaxy_sources/leda_82678.json) | spiral | 132.3558 | 425.360 | 0.02997 | 0.442 | 0.637 | 43.00 |
| [Z 213-14](../data/galaxy_sources/z_213_14.json) | spiral | 132.3558 | 425.360 | 0.02997 | 0.497 | 0.922 | 53.00 |
| [Z 149-45](../data/galaxy_sources/z_149_45.json) | spiral | 132.3997 | 425.499 | 0.02998 | 0.631 | 0.515 | 14.00 |
| [LEDA 32789](../data/galaxy_sources/leda_32789.json) | spiral | 132.4433 | 425.636 | 0.02998994 | 0.471 | 0.535 | 113.00 |
| [MCG+05-26-036a](../data/galaxy_sources/mcg_05_26_036a.json) | spiral | 132.4435 | 425.637 | 0.02999 | 0.508 | 0.606 | 86.00 |
| [MCG+07-18-012](../data/galaxy_sources/mcg_07_18_012.json) | lenticular | 132.4874 | 425.776 | 0.03 | 0.497 | 0.872 | 41.00 |
| [MCG+09-21-083](../data/galaxy_sources/mcg_09_21_083.json) | spiral | 132.5312 | 425.915 | 0.03001 | 0.568 | 0.630 | 7.00 |
| [LEDA 1141211](../data/galaxy_sources/leda_1141211.json) | lenticular | 132.5312 | 425.915 | 0.03001 | 0.696 | 0.334 | 167.00 |
| [MCG+00-54-023](../data/galaxy_sources/mcg_00_54_023.json) | spiral | 132.6189 | 426.193 | 0.03003 | 0.652 | 0.491 | 141.00 |
| [Z 64-56](../data/galaxy_sources/z_64_56.json) | spiral | 132.6189 | 426.193 | 0.03003 | 0.418 | 0.685 | 62.00 |
| [Z 40-19](../data/galaxy_sources/z_40_19.json) | barred spiral | 132.6189 | 426.193 | 0.03003 | 0.483 | 0.750 | 19.00 |
| [Z 212-10](../data/galaxy_sources/z_212_10.json) | spiral | 132.6628 | 426.332 | 0.03004 | 0.548 | 0.450 | 150.00 |
| [LEDA 66575](../data/galaxy_sources/leda_66575.json) | spiral | 132.6845 | 426.400 | 0.03004496 | 0.550 | 0.539 | 3.00 |
| [Z 289-6](../data/galaxy_sources/z_289_6.json) | elliptical | 132.7066 | 426.470 | 0.03005 | 0.732 | 0.875 | 105.00 |
| [MCG+09-21-070](../data/galaxy_sources/mcg_09_21_070.json) | spiral | 132.7066 | 426.470 | 0.03005 | 0.682 | 0.677 | 68.00 |
| [Z 212-52](../data/galaxy_sources/z_212_52.json) | elliptical | 132.7066 | 426.470 | 0.03005 | 0.607 | 0.885 | 81.00 |
| [Z 107-30](../data/galaxy_sources/z_107_30.json) | spiral | 132.7066 | 426.470 | 0.03005 | 0.608 | 0.649 | 119.00 |
| [MCG+09-18-068](../data/galaxy_sources/mcg_09_18_068.json) | barred spiral | 132.7504 | 426.609 | 0.03006 | 0.470 | 0.837 | 28.00 |
| [Z 71-29](../data/galaxy_sources/z_71_29.json) | spiral | 132.7504 | 426.609 | 0.03006 | 0.831 | 0.881 | 44.00 |
| [Z 76-132](../data/galaxy_sources/z_76_132.json) | barred spiral | 132.7504 | 426.609 | 0.03006 | 0.709 | 0.626 | 158.00 |
| [Z 295-18](../data/galaxy_sources/z_295_18.json) | spiral | 132.7943 | 426.748 | 0.03007 | 0.686 | 0.453 | 143.00 |
| [MCG+09-22-058](../data/galaxy_sources/mcg_09_22_058.json) | spiral | 132.7943 | 426.748 | 0.03007 | 0.675 | 0.368 | 118.00 |
| [Z 104-47](../data/galaxy_sources/z_104_47.json) | barred spiral | 132.7943 | 426.748 | 0.03007 | 0.633 | 0.729 | 167.00 |
| [Z 77-88](../data/galaxy_sources/z_77_88.json) | spiral | 132.8820 | 427.026 | 0.03009 | 1.207 | 0.720 | 83.00 |
| [MCG+09-29-006](../data/galaxy_sources/mcg_09_29_006.json) | spiral | 132.9696 | 427.303 | 0.03011 | 0.601 | 0.790 | 169.00 |
| [LEDA 3083145](../data/galaxy_sources/leda_3083145.json) | spiral | 132.9696 | 427.303 | 0.03011 | 0.660 | 0.421 | 175.00 |
| [LEDA 1166530](../data/galaxy_sources/leda_1166530.json) | spiral | 133.0135 | 427.442 | 0.03012 | 0.892 | 0.234 | 39.00 |
| [Z 270-39](../data/galaxy_sources/z_270_39.json) | spiral | 133.0573 | 427.581 | 0.03013 | 1.047 | 0.541 | 143.00 |
| [Z 133-91](../data/galaxy_sources/z_133_91.json) | elliptical | 133.0573 | 427.581 | 0.03013 | 0.640 | 0.738 | 17.00 |
| [Z 294-46](../data/galaxy_sources/z_294_46.json) | spiral | 133.1012 | 427.720 | 0.03014 | 0.670 | 0.857 | 6.00 |
| [Z 179-2](../data/galaxy_sources/z_179_2.json) | spiral | 133.1012 | 427.720 | 0.03014 | 0.597 | 0.495 | 9.00 |
| [LEDA 24447](../data/galaxy_sources/leda_24447.json) | lenticular | 133.1450 | 427.858 | 0.03015 | 0.386 | 0.750 | 53.00 |
| [MCG+10-14-037](../data/galaxy_sources/mcg_10_14_037.json) | barred spiral | 133.1888 | 427.997 | 0.03016 | 0.515 | 0.495 | 171.00 |
| [IC 1391](../data/galaxy_sources/ic_1391.json) | elliptical | 133.1888 | 427.997 | 0.03016 | 0.645 | 0.808 | 86.00 |
| [MCG+10-19-090](../data/galaxy_sources/mcg_10_19_090.json) | spiral | 133.2327 | 428.136 | 0.03017 | 1.041 | 0.186 | 65.00 |
| [Z 164-44](../data/galaxy_sources/z_164_44.json) | spiral | 133.2327 | 428.136 | 0.03017 | 0.677 | 0.656 | 67.00 |
| [IC 1388](../data/galaxy_sources/ic_1388.json) | elliptical | 133.2765 | 428.275 | 0.03018 | 1.027 | 0.537 | 137.00 |
| [SDSS J143138.48-012104.8](../data/galaxy_sources/sdss_j143138_48_012104_8.json) | elliptical | 133.2765 | 428.275 | 0.03018 | 0.177 | 0.780 | 45.00 |
| [MCG+09-22-066](../data/galaxy_sources/mcg_09_22_066.json) | spiral | 133.3204 | 428.414 | 0.03019 | 0.594 | 0.444 | 86.00 |
| [Z 170-9](../data/galaxy_sources/z_170_9.json) | spiral | 133.3204 | 428.414 | 0.03019 | 0.809 | 0.466 | 127.00 |
| [Z 407-37](../data/galaxy_sources/z_407_37.json) | spiral | 133.3630 | 428.549 | 0.03019972 | 1.034 | 0.528 | 19.00 |
| [Z 76-82](../data/galaxy_sources/z_76_82.json) | elliptical | 133.4519 | 428.830 | 0.03022 | 0.762 | 0.628 | 153.00 |
| [Z 165-13](../data/galaxy_sources/z_165_13.json) | spiral | 133.4519 | 428.830 | 0.03022 | 0.510 | 0.947 | 4.00 |
| [LEDA 174039](../data/galaxy_sources/leda_174039.json) | spiral | 133.4519 | 428.830 | 0.03022 | 0.901 | 0.181 | 167.00 |
| [ESO 444-16](../data/galaxy_sources/eso_444_16.json) | spiral | 133.4836 | 428.930 | 0.03022724 | 0.607 | 0.779 | 95.00 |
| [MCG+08-17-081](../data/galaxy_sources/mcg_08_17_081.json) | spiral | 133.4957 | 428.969 | 0.03023 | 0.838 | 0.517 | 35.00 |
| [Z 243-56](../data/galaxy_sources/z_243_56.json) | elliptical | 133.4957 | 428.969 | 0.03023 | 0.708 | 0.750 | 94.00 |
| [Z 94-118](../data/galaxy_sources/z_94_118.json) | spiral | 133.4957 | 428.969 | 0.03023 | 0.887 | 0.284 | 176.00 |
| [Z 63-95](../data/galaxy_sources/z_63_95.json) | spiral | 133.5396 | 429.108 | 0.03024 | 0.617 | 0.460 | 101.00 |
| [Z 374-26](../data/galaxy_sources/z_374_26.json) | spiral | 133.5396 | 429.108 | 0.03024 | 0.618 | 0.808 | 60.00 |
| [Z 139-4](../data/galaxy_sources/z_139_4.json) | spiral | 133.5834 | 429.246 | 0.03025 | 0.715 | 0.592 | 17.00 |
| [MCG+09-20-164](../data/galaxy_sources/mcg_09_20_164.json) | lenticular | 133.6272 | 429.385 | 0.03026 | 0.599 | 0.407 | 137.00 |
| [UGC 8706](../data/galaxy_sources/ugc_8706.json) | spiral | 133.6272 | 429.385 | 0.03026 | 0.833 | 0.181 | 174.00 |
| [Z 165-30](../data/galaxy_sources/z_165_30.json) | spiral | 133.6272 | 429.385 | 0.03026 | 0.550 | 0.463 | 58.00 |
| [LEDA 70136](../data/galaxy_sources/leda_70136.json) | spiral | 133.6272 | 429.385 | 0.03026 | 0.645 | 0.372 | 177.00 |
| [Z 75-28](../data/galaxy_sources/z_75_28.json) | spiral | 133.7098 | 429.646 | 0.03027883 | 0.540 | 0.799 | 44.00 |
| [Z 164-52](../data/galaxy_sources/z_164_52.json) | spiral | 133.7588 | 429.802 | 0.03029 | 0.608 | 0.542 | 34.00 |
| [Z 67-3](../data/galaxy_sources/z_67_3.json) | spiral | 133.8026 | 429.940 | 0.0303 | 0.349 | 0.808 | 33.00 |
| [LEDA 188720](../data/galaxy_sources/leda_188720.json) | spiral | 133.8026 | 429.940 | 0.0303 | 0.406 | 0.591 | 136.00 |
| [MCG+08-23-063](../data/galaxy_sources/mcg_08_23_063.json) | lenticular | 133.8464 | 430.079 | 0.03031 | 0.704 | 0.487 | 168.00 |
| [Z 35-81](../data/galaxy_sources/z_35_81.json) | lenticular | 133.8464 | 430.079 | 0.03031 | 0.713 | 0.453 | 137.00 |
| [Z 36-11](../data/galaxy_sources/z_36_11.json) | lenticular | 133.8903 | 430.218 | 0.03032 | 0.434 | 0.828 | 5.00 |
| [IC 3714](../data/galaxy_sources/ic_3714.json) | spiral | 133.8903 | 430.218 | 0.03032 | 0.677 | 0.634 | 151.00 |
| [Z 164-51](../data/galaxy_sources/z_164_51.json) | spiral | 133.8903 | 430.218 | 0.03032 | 0.634 | 0.872 | 36.00 |
| [Z 165-18](../data/galaxy_sources/z_165_18.json) | lenticular | 133.8903 | 430.218 | 0.03032 | 0.488 | 0.711 | 88.00 |
| [Z 428-4](../data/galaxy_sources/z_428_4.json) | spiral | 133.9341 | 430.357 | 0.03033 | 0.874 | 0.555 | 104.00 |
| [NGC 3700](../data/galaxy_sources/ngc_3700.json) | barred spiral | 133.9779 | 430.495 | 0.03034 | 0.924 | 0.810 | 174.00 |
| [Z 212-32](../data/galaxy_sources/z_212_32.json) | spiral | 134.0656 | 430.773 | 0.03036 | 1.046 | 0.515 | 77.00 |
| [Z 138-71](../data/galaxy_sources/z_138_71.json) | spiral | 134.1094 | 430.912 | 0.03037 | 0.726 | 0.327 | 38.00 |
| [LEDA 24339](../data/galaxy_sources/leda_24339.json) | spiral | 134.1533 | 431.050 | 0.03038 | 0.512 | 0.604 | 114.00 |
| [Z 60-27](../data/galaxy_sources/z_60_27.json) | spiral | 134.1971 | 431.189 | 0.03039 | 0.843 | 0.665 | 159.00 |
| [MCG+10-24-103](../data/galaxy_sources/mcg_10_24_103.json) | spiral | 134.2410 | 431.328 | 0.0304 | 0.624 | 0.285 | 42.00 |
| [Z 19-63](../data/galaxy_sources/z_19_63.json) | elliptical | 134.2410 | 431.328 | 0.0304 | 1.308 | 0.688 | 66.00 |
| [Z 192-53](../data/galaxy_sources/z_192_53.json) | spiral | 134.2848 | 431.467 | 0.03041 | 0.591 | 0.947 | 179.00 |
| [MCG+03-04-025](../data/galaxy_sources/mcg_03_04_025.json) | spiral | 134.3130 | 431.556 | 0.03041643 | 0.503 | 0.971 | 21.00 |
| [Z 72-41](../data/galaxy_sources/z_72_41.json) | elliptical | 134.3725 | 431.744 | 0.03043 | 0.895 | 0.772 | 83.00 |
| [LEDA 987357](../data/galaxy_sources/leda_987357.json) | spiral | 134.3725 | 431.744 | 0.03043 | 1.009 | 0.237 | 147.00 |
| [LEDA 79630](../data/galaxy_sources/leda_79630.json) | spiral | 134.4601 | 432.022 | 0.03045 | 0.583 | 0.599 | 84.00 |
| [Z 430-37](../data/galaxy_sources/z_430_37.json) | spiral | 134.5040 | 432.160 | 0.03046 | 0.577 | 0.407 | 78.00 |
| [Z 263-61](../data/galaxy_sources/z_263_61.json) | elliptical | 134.5478 | 432.299 | 0.03047 | 0.967 | 0.773 | 56.00 |
| [Z 273-7](../data/galaxy_sources/z_273_7.json) | spiral | 134.5478 | 432.299 | 0.03047 | 0.699 | 0.603 | 146.00 |
| [LEDA 24366](../data/galaxy_sources/leda_24366.json) | lenticular | 134.5478 | 432.299 | 0.03047 | 0.375 | 0.825 | 48.00 |
| [Z 195-14](../data/galaxy_sources/z_195_14.json) | spiral | 134.5478 | 432.299 | 0.03047 | 0.882 | 0.315 | 146.00 |
| [LEDA 1125446](../data/galaxy_sources/leda_1125446.json) | lenticular | 134.5478 | 432.299 | 0.03047 | 0.509 | 0.476 | 104.00 |
| [MCG+08-27-025](../data/galaxy_sources/mcg_08_27_025.json) | spiral | 134.5916 | 432.438 | 0.03048 | 0.742 | 0.601 | 78.00 |
| [Z 163-17](../data/galaxy_sources/z_163_17.json) | spiral | 134.5916 | 432.438 | 0.03048 | 0.699 | 0.674 | 162.00 |
| [IC 1052](../data/galaxy_sources/ic_1052.json) | barred spiral | 134.5916 | 432.438 | 0.03048 | 0.914 | 0.324 | 27.00 |
| [MCG+03-38-071](../data/galaxy_sources/mcg_03_38_071.json) | barred spiral | 134.5916 | 432.438 | 0.03048 | 0.666 | 0.755 | 164.00 |
| [LEDA 3440788](../data/galaxy_sources/leda_3440788.json) | lenticular | 134.5916 | 432.438 | 0.03048 | 0.510 | 0.523 | 2.00 |
| [NGC 6295](../data/galaxy_sources/ngc_6295.json) | spiral | 134.6355 | 432.577 | 0.03049 | 1.136 | 0.471 | 75.00 |
| [Z 269-60](../data/galaxy_sources/z_269_60.json) | spiral | 134.6355 | 432.577 | 0.03049 | 0.539 | 0.651 | 76.00 |
| [Z 58-18](../data/galaxy_sources/z_58_18.json) | spiral | 134.6447 | 432.606 | 0.03049211 | 0.605 | 0.740 | 1.00 |
| [Z 427-21](../data/galaxy_sources/z_427_21.json) | spiral | 134.6749 | 432.701 | 0.030499 | 0.655 | 0.535 | 28.00 |
| [LEDA 89698](../data/galaxy_sources/leda_89698.json) | spiral | 134.6793 | 432.715 | 0.0305 | 0.193 | 0.902 | 90.00 |
| [MCG+09-20-093](../data/galaxy_sources/mcg_09_20_093.json) | spiral | 134.7670 | 432.993 | 0.03052 | 0.691 | 0.387 | 18.00 |
| [MCG+08-23-053](../data/galaxy_sources/mcg_08_23_053.json) | spiral | 134.7670 | 432.993 | 0.03052 | 1.642 | 0.249 | 177.00 |
| [IC 1786](../data/galaxy_sources/ic_1786.json) | spiral | 134.7805 | 433.036 | 0.03052308 | 0.639 | 0.643 | 155.00 |
| [Z 169-49](../data/galaxy_sources/z_169_49.json) | barred spiral | 134.8108 | 433.132 | 0.03053 | 0.865 | 0.333 | 130.00 |
| [Z 314-15](../data/galaxy_sources/z_314_15.json) | spiral | 134.8546 | 433.270 | 0.03054 | 0.658 | 0.556 | 135.00 |
| [Z 64-74](../data/galaxy_sources/z_64_74.json) | elliptical | 134.8985 | 433.409 | 0.03055 | 0.842 | 0.688 | 10.00 |
| [MCG+08-23-086](../data/galaxy_sources/mcg_08_23_086.json) | spiral | 134.9423 | 433.548 | 0.03056 | 0.613 | 0.650 | 99.00 |
| [UGC 884](../data/galaxy_sources/ugc_884.json) | spiral | 135.0067 | 433.751 | 0.03057469 | 0.700 | 0.719 | 179.00 |
| [NGC 6157](../data/galaxy_sources/ngc_6157.json) | elliptical | 135.0299 | 433.825 | 0.03058 | 0.790 | 0.744 | 179.00 |
| [Z 390-97](../data/galaxy_sources/z_390_97.json) | spiral | 135.0299 | 433.825 | 0.03058 | 0.543 | 0.895 | 83.00 |
| [Z 156-88](../data/galaxy_sources/z_156_88.json) | lenticular | 135.0299 | 433.825 | 0.03058 | 0.938 | 0.299 | 101.00 |
| [LEDA 39735](../data/galaxy_sources/leda_39735.json) | spiral | 135.0299 | 433.825 | 0.03058 | 0.422 | 0.403 | 86.00 |
| [IC 1037](../data/galaxy_sources/ic_1037.json) | elliptical | 135.0299 | 433.825 | 0.03058 | 1.048 | 0.699 | 108.00 |
| [Z 77-32](../data/galaxy_sources/z_77_32.json) | spiral | 135.0299 | 433.825 | 0.03058 | 0.463 | 0.320 | 15.00 |
| [Z 277-42](../data/galaxy_sources/z_277_42.json) | spiral | 135.0738 | 433.964 | 0.03059 | 0.853 | 0.683 | 175.00 |
| [LEDA 66795](../data/galaxy_sources/leda_66795.json) | lenticular | 135.0738 | 433.964 | 0.03059 | 0.416 | 0.934 | 5.00 |
| [Z 295-19](../data/galaxy_sources/z_295_19.json) | spiral | 135.1176 | 434.103 | 0.0306 | 0.954 | 0.419 | 138.00 |
| [Z 49-177](../data/galaxy_sources/z_49_177.json) | lenticular | 135.1176 | 434.103 | 0.0306 | 0.451 | 0.595 | 90.00 |
| [IC 2930](../data/galaxy_sources/ic_2930.json) | spiral | 135.1614 | 434.241 | 0.03061 | 0.678 | 0.372 | 60.00 |
| [Z 407-16](../data/galaxy_sources/z_407_16.json) | spiral | 135.2027 | 434.372 | 0.03061942 | 0.636 | 0.812 | 61.00 |
| [MCG+03-08-037](../data/galaxy_sources/mcg_03_08_037.json) | spiral | 135.2027 | 434.372 | 0.03061942 | 0.270 | 0.541 | 115.00 |
