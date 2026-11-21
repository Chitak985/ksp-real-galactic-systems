# Fifteen reviewed nearby galaxies and forty-two distant galaxies

This batch adds exactly **57** real objects: all **42** remaining eligible records
beyond 150 Mpc in the saved bulk sample, plus the **15** nearby candidates from the
location audit. The generated catalog now contains **2,057** galaxies, or **2,060**
disk definitions including the three legacy galaxy bodies. The previous 2,000
galaxies, visual refinements, source snapshots, all stars, Sun and README are
preserved.

## Research and adoption policy

The review was performed on 2026-11-20 using discovery papers, recent HST analyses,
spectroscopic work and the saved LVDB v1.1.2 data. Per-object primary links, notes,
measurement updates and velocity policies are stored in
`data/reviewed_nearby_candidates.json`. Original LVDB rows remain unchanged in
each new raw record; the review is a separate block applied by the normalizer.
This preserves the distinction between compiled data and later adoptions.

Geometry and placement do not require a radial velocity when an independent
distance is available. Thirteen new objects therefore retain `redshift: null`
and `redshift_status: unmeasured`. Their colors use an unshifted display palette,
and descriptions explicitly state that approximation. No zero velocity is saved
as a measurement, and no redshift-derived distance is used for these objects.

Two explicit proxies are adopted:

- **IKN:** the IKN4/5 integrated cluster spectrum gives 38 +/- 30 km/s. This is
  labeled `cluster_proxy`; other cluster spectra differ, so it is not treated as
  a secure galaxy systemic velocity. A later GC study also identifies a separate
  projected candidate as a background object. [Cluster study](https://arxiv.org/abs/2406.01279),
  [later spectroscopy](https://arxiv.org/html/2405.11749v1).
- **Centaurus C:** the SEAMLESS paper's appendix gives a CosmicFlows-4 prediction
  of 300 km/s. This is labeled `model_proxy`, with no invented error bar. Its
  independent TRGB distance still controls placement.
  [SEAMLESS analysis](https://arxiv.org/html/2610.40765v1).

Missing or proxy redshifts are permitted only with explicit status and an
independent/adopted local distance. A cosmological distance cannot use them.
Default strict imports still reject absent velocities without a reviewed policy.
Reviews cannot change confirmation flags, object identities or catalog tables,
or silently replace an existing velocity measurement.

## Nearby additions and research outcomes

| Galaxy | Adopted distance (Mpc) | Velocity status | Review outcome and primary source |
|---|---:|---|---|
| [Carina IV](../data/galaxy_sources/lvdb_carina_4.json) | 0.10471 | unmeasured | Photometric dwarf candidate; kinematics and LMC association uncertain. [Paper](https://doi.org/10.3847/1538-4357/ae43e2) |
| [Andromeda XXXV](../data/galaxy_sources/lvdb_andromeda_35.json) | 0.92470 | unmeasured | Sparse stellar system with measured photometric distance and size. [Paper](https://arxiv.org/abs/2502.19516) |
| [Andromeda XXXVI](../data/galaxy_sources/lvdb_andromeda_36.json) | 0.77600 | unmeasured | M31 distance assumed; classification and near-round geometry need follow-up. [Paper](https://arxiv.org/html/2603.28492v1) |
| [Pegasus VII](../data/galaxy_sources/lvdb_pegasus_7.json) | 0.75162 | unmeasured | Photometric elongation does not establish an orbit or tidal history. [Paper](https://arxiv.org/html/2502.09792v1) |
| [Pisces VII](../data/galaxy_sources/lvdb_pisces_7.json) | 0.92897 | unmeasured | HST distance retained; possible M33 association remains uncertain. [Paper](https://arxiv.org/html/2508.00984v2) |
| [Leo K](../data/galaxy_sources/lvdb_leo_k.json) | 0.43400 | unmeasured | Distance 434 +17/-127 kpc; population-fit stellar mass adopted. [Paper](https://arxiv.org/html/2307.08738v2) |
| [Leo M](../data/galaxy_sources/lvdb_leo_m.json) | 0.45900 | unmeasured | Distance 459 +21/-18 kpc; population-fit stellar mass adopted. [Paper](https://arxiv.org/html/2307.08738v2) |
| [Sculptor A](../data/galaxy_sources/lvdb_sculptor_a.json) | 1.35000 | unmeasured | Published asymmetric distance errors and population mass retained. [Paper](https://arxiv.org/html/2409.16345v2) |
| [Sculptor B](../data/galaxy_sources/lvdb_sculptor_b.json) | 2.48000 | unmeasured | Distance uncertainty affects its relationship with NGC 300. [Paper](https://arxiv.org/html/2409.16345v2) |
| [Sculptor C](../data/galaxy_sources/lvdb_sculptor_c.json) | 2.04000 | unmeasured | NGC 300 association is consistent with its photometric distance. [Paper](https://arxiv.org/html/2409.16345v2) |
| [Centaurus C](../data/galaxy_sources/lvdb_centaurus_c.json) | 4.05000 | model proxy | TRGB error and SFH mass adopted; flow prediction labeled. [Paper](https://arxiv.org/html/2610.40765v1) |
| [Donatiello VIII](../data/galaxy_sources/lvdb_donatiello_8.json) | 3.63000 | unmeasured | Cetus B alias confirmed; HST shape replaces older integrated-light geometry. [Paper](https://arxiv.org/html/2610.40765v1) |
| [Hydrus A](../data/galaxy_sources/lvdb_hydrus_a.json) | 4.04000 | unmeasured | HST distance uncertainty and population-fit mass adopted. [Paper](https://arxiv.org/html/2610.40765v1) |
| [IKN](../data/galaxy_sources/lvdb_ikn.json) | 3.74973 | cluster proxy | Cluster proxy; luminosity and stellar mass remain uncertain. [Paper](https://arxiv.org/abs/2406.01279) |
| [Bedin 1](../data/galaxy_sources/lvdb_bedin_1.json) | 8.70000 | unmeasured | TRGB errors restored; V approximates F606W; foreground contamination uncertain. [Paper](https://arxiv.org/abs/1902.00271) |


The new nearby profiles remain Gaussian approximations derived from their
measured half-light sizes, with measured apparent flattening and position angle.
Three-dimensional depth and detailed stellar distributions are uncertain.
Published population-fit stellar masses replace generic light-to-mass estimates
where reviewed; these remain model estimates excluding dark halos. Other objects
retain the existing V-band mass/light convention. All descriptions include the
applicable uncertainties.

The seven previously identified Local Group candidates are now included, giving
a working Local Group count of 94. Andromeda XXXVI remains a photometric candidate
at an assumed M31 distance. Sculptor A lies near the Local Group boundary and is
kept in the nearby external sample for this accounting; that boundary assignment
is uncertain. The remaining eight additions cover the nearby external volume.

## Distant sample

These 42 entries retain the existing SIMBAD quality gates, measured dimensions,
sky orientation, B/K photometry, spectroscopic redshift, primary references and
FLRW distance conversion. Selection is strictly greater than 150 Mpc in nominal
comoving placement distance, extending to 150.752 Mpc. This is all eligible unused
data beyond that threshold in the saved continuation snapshot; no completeness
claim is made about the real universe. The 655 eligible records below the
threshold were subsequently added in the [100-150 Mpc batch](Remaining100To150MpcGalaxies.md).

Each raw record links to the existing immutable SIMBAD response snapshot. The
complete selected IDs and source rejections are saved separately in
`data/far_near_galaxy_selection.json`, preserving earlier batch audits.

| Galaxy / source | Template | Comoving placement (Mpc) | Lookback (Myr) | z |
|---|---|---:|---:|---:|
| [Z 152-66](../data/galaxy_sources/z_152_66.json) | lenticular | 150.00773 | 481.148 | 0.034 |
| [Z 503-42](../data/galaxy_sources/z_503_42.json) | spiral | 150.04157 | 481.255 | 0.03400773 |
| [LEDA 27289](../data/galaxy_sources/leda_27289.json) | barred spiral | 150.05149 | 481.286 | 0.03401 |
| [Z 433-35](../data/galaxy_sources/z_433_35.json) | spiral | 150.05149 | 481.286 | 0.03401 |
| [Z 78-54](../data/galaxy_sources/z_78_54.json) | spiral | 150.05149 | 481.286 | 0.03401 |
| [LEDA 56860](../data/galaxy_sources/leda_56860.json) | spiral | 150.05149 | 481.286 | 0.03401 |
| [MCG+09-20-165](../data/galaxy_sources/mcg_09_20_165.json) | spiral | 150.09525 | 481.424 | 0.03402 |
| [MCG-02-04-025](../data/galaxy_sources/mcg_02_04_025.json) | spiral | 150.09525 | 481.424 | 0.03402 |
| [Z 272-14](../data/galaxy_sources/z_272_14.json) | spiral | 150.13901 | 481.562 | 0.03403 |
| [Z 122-47](../data/galaxy_sources/z_122_47.json) | spiral | 150.18276 | 481.700 | 0.03404 |
| [UGC 12612](../data/galaxy_sources/ugc_12612.json) | spiral | 150.26822 | 481.970 | 0.03405953 |
| [Z 270-35](../data/galaxy_sources/z_270_35.json) | elliptical | 150.27028 | 481.976 | 0.03406 |
| [LEDA 79525](../data/galaxy_sources/leda_79525.json) | spiral | 150.27028 | 481.976 | 0.03406 |
| [LEDA 92988](../data/galaxy_sources/leda_92988.json) | spiral | 150.27028 | 481.976 | 0.03406 |
| [MCG+08-17-040](../data/galaxy_sources/mcg_08_17_040.json) | spiral | 150.31403 | 482.114 | 0.03407 |
| [Z 119-13](../data/galaxy_sources/z_119_13.json) | spiral | 150.31403 | 482.114 | 0.03407 |
| [Z 437-6](../data/galaxy_sources/z_437_6.json) | elliptical | 150.31403 | 482.114 | 0.03407 |
| [MCG+07-23-026](../data/galaxy_sources/mcg_07_23_026.json) | spiral | 150.31403 | 482.114 | 0.03407 |
| [Z 242-62](../data/galaxy_sources/z_242_62.json) | spiral | 150.31403 | 482.114 | 0.03407 |
| [UGC 12384](../data/galaxy_sources/ugc_12384.json) | barred spiral | 150.37398 | 482.303 | 0.0340837 |
| [LEDA 1165719](../data/galaxy_sources/leda_1165719.json) | spiral | 150.40420 | 482.399 | 0.03409061 |
| [Z 72-28](../data/galaxy_sources/z_72_28.json) | spiral | 150.44530 | 482.528 | 0.0341 |
| [Z 109-4](../data/galaxy_sources/z_109_4.json) | elliptical | 150.44530 | 482.528 | 0.0341 |
| [Z 19-61](../data/galaxy_sources/z_19_61.json) | lenticular | 150.48906 | 482.666 | 0.03411 |
| [Z 503-48](../data/galaxy_sources/z_503_48.json) | spiral | 150.49487 | 482.685 | 0.03411133 |
| [Z 480-33](../data/galaxy_sources/z_480_33.json) | spiral | 150.50998 | 482.732 | 0.03411478 |
| [MCG+07-23-041](../data/galaxy_sources/mcg_07_23_041.json) | spiral | 150.53281 | 482.804 | 0.03412 |
| [LEDA 1614](../data/galaxy_sources/leda_1614.json) | elliptical | 150.55531 | 482.875 | 0.03412514 |
| [LEDA 70340](../data/galaxy_sources/leda_70340.json) | spiral | 150.57657 | 482.942 | 0.03413 |
| [SDSS J132005.64+331713.3](../data/galaxy_sources/sdss_j132005_64_331713_3.json) | elliptical | 150.57657 | 482.942 | 0.03413 |
| [Z 225-18](../data/galaxy_sources/z_225_18.json) | elliptical | 150.57657 | 482.942 | 0.03413 |
| [UGC 12617](../data/galaxy_sources/ugc_12617.json) | spiral | 150.61575 | 483.066 | 0.03413895 |
| [MCG+03-33-030](../data/galaxy_sources/mcg_03_33_030.json) | spiral | 150.62032 | 483.080 | 0.03414 |
| [LEDA 89641](../data/galaxy_sources/leda_89641.json) | spiral | 150.66408 | 483.218 | 0.03415 |
| [MCG+10-17-094](../data/galaxy_sources/mcg_10_17_094.json) | spiral | 150.70783 | 483.356 | 0.03416 |
| [MCG+07-23-004](../data/galaxy_sources/mcg_07_23_004.json) | spiral | 150.70783 | 483.356 | 0.03416 |
| [Z 109-24](../data/galaxy_sources/z_109_24.json) | barred spiral | 150.70783 | 483.356 | 0.03416 |
| [Z 136-1](../data/galaxy_sources/z_136_1.json) | spiral | 150.70783 | 483.356 | 0.03416 |
| [LEDA 1175041](../data/galaxy_sources/leda_1175041.json) | spiral | 150.70783 | 483.356 | 0.03416 |
| [Z 477-9](../data/galaxy_sources/z_477_9.json) | spiral | 150.72152 | 483.399 | 0.03416313 |
| [MCG+09-20-147](../data/galaxy_sources/mcg_09_20_147.json) | spiral | 150.75159 | 483.494 | 0.03417 |
| [LEDA 87009](../data/galaxy_sources/leda_87009.json) | spiral | 150.75159 | 483.494 | 0.03417 |


## Rebuilding and replay

```powershell
python rebuild_galaxies.py
python rebuild_galaxies.py --check
python add_reviewed_galaxies.py --dry-run
```

`add_reviewed_galaxies.py` uses saved data and the explicit reviews, imports only
new objects, and selects every unused eligible bulk record beyond 150 Mpc. It
does not query the network or refresh existing objects. Repeating it after this
batch reports that all candidates are already present. Replay the original
selection in a catalog copy with these 57 entries removed. Templates remain the
source for rebuilding; the source rows and research adoptions are stored as data.

All new barycenters use the existing hidden circular Sun-parented orbit with
`period = 3E300`, `mode = OFF`, static disk rotation, small 5-100 pc core SOIs and
2,000 representative mesh stars. Configs are directly under
`RealGalacticSystems/Systems`. No new core SOI overlaps were found among the
2,057 generated galaxy barycenters. Live KSP resource use remains unmeasured.


## Verification

All 69 automated tests pass. New coverage checks all 57 requested IDs, the
strict distant boundary, preservation of original rows, reviewed source replay,
null unknown redshifts, labeled velocity proxies, published uncertainties,
assumed host distance, resolved-star geometry, confirmation safeguards and
write-free offline selection replay. Existing source, orientation, cosmology,
template propagation and protection checks still pass. All 2,057 generated galaxy
configs and all 150 generated star configs match their data and templates.
Orientation checks cover 2,060 GalacticDisk definitions.

All fifteen nearby objects and eleven distant representatives were rendered and
visually inspected using the current local GalacticDisk C# source and matching
packaged shaders in isolated Unity 2019.4. Views use the Sol-to-galaxy sky direction
with enlarged framing. These checks use KSP API substitutes, do not open saves,
and do not establish absolute photometric accuracy or full-catalog game
performance. Existing configs and source files were compared byte for byte.
