# Stars between the nearby and outer-disk batches

This batch adds **60 real catalogued Sun-like candidates**, bringing the saved catalog to **150** stars. Ten are selected in each of six distance bands to bridge the nearby sample and the earlier outer-disk candidates. Researched on 2026-10-24. The entries use the existing single-star template and Sun-parented frozen positioning orbits.

The additions span **139-12,669 light-years from Sol**, with estimated Galactic radii **8.23-11.51 kpc** (adopting R0=8.2 kpc). This fills the catalog's outward gap; it is not a claim to sample the Galaxy's central inner disk. Galactic longitude is roughly 120-240 degrees. Gaia directions are used consistently for placement and region calculations.

## Distance coverage

| Band | Target distance from Sol (kpc) | Actual selected distance (ly) | Stars | Physical catalog |
| --- | ---: | ---: | ---: | --- |
| 1 | 0.03-0.15 | 139-453 | 10 | Gaia RVS DR3 |
| 2 | 0.15-0.4 | 513-1,240 | 10 | Gaia RVS DR3 |
| 3 | 0.4-0.8 | 1,347-1,964 | 10 | Gaia RVS DR3 |
| 4 | 0.8-1.5 | 2,684-4,746 | 10 | LAMOST LRS DR7 |
| 5 | 1.5-2.5 | 4,967-7,704 | 10 | LAMOST LRS DR7 |
| 6 | 2.5-4 | 8,331-12,669 | 10 | LAMOST LRS DR7 |

Within each band, candidates were chosen near logarithmically spaced distance targets, favoring different sky directions and narrower distance posteriors. All 60 have distinct Gaia IDs and differ from the existing 90 records. Known SIMBAD names are used when available; otherwise the Gaia DR3 designation identifies the star. A Gaia ID remains the stable catalog key.

## Measurements, estimates and uncertainties

The corrected [StarHorse catalogs of Queiroz et al. (2023)](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A+A/673/A155?format=html&tex=true) provide posterior mass, temperature, gravity, metallicity, distance and age estimates. This batch uses 30 Gaia RVS and 30 LAMOST LRS entries. Radius is derived from R=sqrt(G*M/g); luminosity uses L/Lsun=(R/Rsun)**2*(T/5772 K)**4. Reported values, derived values, original quality flags, posterior percentiles and sensitivity ranges are saved separately in `data/bodies.json`.

59 spin periods are classical gyrochronology estimates using [Barnes (2007)](https://arxiv.org/abs/0704.3068) and the temperature/color approximation of [Ballesteros (2012)](https://arxiv.org/abs/1201.1809), as in the previous batch. The input-sensitivity ranges are not calibrated prediction intervals; model age, metallicity, differential rotation and old-star [weakened magnetic braking](https://arxiv.org/abs/1601.02631) can bias the adopted spins.

Gaia DR3 2993520260320097152 instead uses the published **5.4431 +/- 0.6254 day** photometric rotation estimate, adopted as **5.44 days**. [Gaia's rotational-modulation analysis](https://arxiv.org/abs/2206.05500) measures variability rather than a unique equatorial rotation. Its StarHorse age would predict a much longer classical spin, so age/spin consistency remains uncertain. The observed estimate is kept rather than replaced by the model.

Every new game description is fewer than 100 characters and briefly states estimated properties and unconfirmed single status. The measured-period star instead notes its uncertain spin. `data_status=estimated_candidate` and `multiplicity_status=no_known_stellar_companion` remain explicit: these are single-star configurations, not proof that all companions have been ruled out. Published distance 5th/95th percentiles describe 90% of the model posterior, not 90% accuracy of every point value.

## Screening and positioning

The physical cuts match the earlier batch: estimated 5300-6000 K, logg 4.2-4.65 with its 16th percentile >=4.0, mass 0.7-1.25 solar, radius 0.7-1.5 solar, metallicity -0.8 to +0.4, a usable MSTO-model age, and no StarHorse output/age-consistency warning. 216 candidates were screened; 148 passed and 60 were selected.

All selected stars have Gaia NSS=0, RUWE<1.2, duplicated_source=false and no multiple-image peak flag. Known SIMBAD multiple/spectroscopic/eclipsing binaries, conflicting evolved or non-G classifications, non-solar Gaia variability, missing astrometry and >3-sigma parallax/distance contradictions were skipped. Near NGC 1981, a cluster-designated source with an old fitted age was skipped for review because the [young-cluster context](https://academic.oup.com/mnras/article/407/3/1875/1070988) creates an evolutionary ambiguity; this is not a new determination of that star's membership.

Directions and proper motions come directly from Gaia DR3. Some StarHorse Gaia-RVS longitude/latitude values differ from the Gaia coordinates; those original values are retained for traceability but not used for placement or the final Galactic radius. Directions are propagated from J2016 to J2000 with the existing tangent-plane method; perspective acceleration is neglected. StarHorse distance medians, not inverse parallax, set the placement distance. No Gaia DSC binary probabilities are used as a certification of singleness. Missing variability or SIMBAD entries are absence of information.

## Added stars

Distances and Galactic radii below are in kiloparsecs; radius, mass and luminosity are in solar units. All physics values are estimates. Spin is modeled except the one observed-modulation entry marked with an asterisk.

| Name | Gaia DR3 source ID | Sol distance | Galactic radius | Radius | Mass | Temperature (K) | Luminosity | Spin (days) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HD 47309 | 992524409046563840 | 0.0426 | 8.239 | 1.1700 | 0.9673 | 5829 | 1.4238 | 36.4 |
| HD 13931 | 351537889154226816 | 0.0471 | 8.234 | 1.2844 | 1.0051 | 5874 | 1.7694 | 33.9 |
| HD 62549 | 3056551894686671616 | 0.0493 | 8.235 | 1.2698 | 1.0156 | 5963 | 1.8366 | 30.3 |
| HD 42661 | 2913272575964428160 | 0.0561 | 8.233 | 1.1310 | 0.8930 | 5753 | 1.2624 | 45.7 |
| HD 25295 | 169256041140206848 | 0.0626 | 8.258 | 1.1868 | 0.9454 | 5929 | 1.5681 | 34.2 |
| HD 261547 | 3356224819628117376 | 0.0681 | 8.264 | 1.2008 | 0.9814 | 5657 | 1.3304 | 42.9 |
| HD 36373 | 3445877692084716160 | 0.0880 | 8.288 | 1.2096 | 1.0264 | 5985 | 1.6914 | 27.7 |
| HD 71938 | 3065665501751684992 | 0.0981 | 8.261 | 1.1477 | 0.9236 | 5926 | 1.4635 | 35.9 |
| BD+67 59 | 529831705422754816 | 0.1017 | 8.254 | 1.1074 | 0.8967 | 5819 | 1.2668 | 42.4 |
| HD 289396 | 3113777038207534464 | 0.1389 | 8.317 | 1.1165 | 0.9091 | 5826 | 1.2939 | 41.6 |
| TYC 2951-903-1 | 951148377744097024 | 0.1574 | 8.348 | 1.1992 | 0.9476 | 5912 | 1.5828 | 35.1 |
| TYC 4344-1193-1 | 484859102862936448 | 0.1728 | 8.333 | 1.2601 | 0.9852 | 5997 | 1.8503 | 30.4 |
| TYC 7073-1089-1 | 2893542904955343616 | 0.1931 | 8.298 | 1.1697 | 0.9679 | 5827 | 1.4211 | 36.5 |
| HD 287360 | 3391557694502418688 | 0.2116 | 8.401 | 1.2252 | 0.9929 | 5844 | 1.5774 | 33.2 |
| TYC 1355-967-1 | 3362726300600063616 | 0.2323 | 8.411 | 1.2380 | 1.1098 | 5987 | 1.7741 | 23.5 |
| TYC 2871-1897-1 | 237471597954470400 | 0.2560 | 8.426 | 1.2151 | 1.0375 | 5782 | 1.4867 | 33.6 |
| Gaia DR3 3047260604817897984 | 3047260604817897984 | 0.2872 | 8.401 | 1.2491 | 1.0379 | 5583 | 1.3657 | 41.3 |
| Gaia DR3 3418857365631186944 | 3418857365631186944 | 0.3470 | 8.543 | 1.3472 | 1.0734 | 5692 | 1.7164 | 37.1 |
| TYC 3253-1300-1 | 390005025008449152 | 0.3498 | 8.379 | 1.1475 | 0.9589 | 5902 | 1.4395 | 34 |
| TYC 166-1582-1 | 3115566390300407552 | 0.3802 | 8.523 | 1.2435 | 0.9846 | 5996 | 1.8007 | 30.5 |
| Gaia DR3 143210358725890432 | 143210358725890432 | 0.4131 | 8.536 | 1.2620 | 1.1180 | 5981 | 1.8362 | 22.5 |
| Gaia DR3 5725340599855025280 | 5725340599855025280 | 0.4439 | 8.468 | 1.2330 | 1.0729 | 5915 | 1.6766 | 27 |
| Gaia DR3 549323499660422784 | 549323499660422784 | 0.4755 | 8.500 | 1.2242 | 1.0133 | 5938 | 1.6787 | 30.4 |
| Gaia DR3 1006126536113569920 | 1006126536113569920 | 0.5247 | 8.643 | 1.3467 | 1.1517 | 5987 | 2.0993 | 22.8 |
| Gaia DR3 388369260942001536 | 388369260942001536 | 0.5377 | 8.473 | 1.3637 | 1.1689 | 5859 | 1.9744 | 25 |
| Gaia DR3 3096192175083401088 | 3096192175083401088 | 0.5443 | 8.625 | 1.3057 | 1.0120 | 5958 | 1.9355 | 30.5 |
| Gaia DR3 509591002031398528 | 509591002031398528 | 0.5459 | 8.554 | 1.3193 | 1.0690 | 5743 | 1.7058 | 36.3 |
| Gaia DR3 2993520260320097152 | 2993520260320097152 | 0.5552 | 8.607 | 1.3423 | 1.0920 | 5958 | 2.0455 | 5.44* |
| Gaia DR3 3399867287991286016 | 3399867287991286016 | 0.5586 | 8.753 | 1.3502 | 1.0585 | 5929 | 2.0296 | 29.4 |
| Gaia DR3 2929037098813808640 | 2929037098813808640 | 0.6023 | 8.564 | 1.2179 | 1.0521 | 5825 | 1.5385 | 30.6 |
| Gaia DR3 387412204788841216 | 387412204788841216 | 0.8229 | 8.621 | 1.2169 | 0.9988 | 5983 | 1.7095 | 29.4 |
| Gaia DR3 3354767623124097664 | 3354767623124097664 | 0.8842 | 9.027 | 1.2754 | 0.9756 | 5744 | 1.5953 | 41.9 |
| Gaia DR3 166764822668271616 | 166764822668271616 | 0.9235 | 9.046 | 1.1707 | 0.9885 | 5920 | 1.5166 | 31.6 |
| Gaia DR3 485668240340941056 | 485668240340941056 | 0.9974 | 8.972 | 1.1455 | 0.8949 | 5991 | 1.5229 | 34.2 |
| Gaia DR3 961569548952685056 | 961569548952685056 | 1.0567 | 9.219 | 1.2193 | 1.0507 | 5758 | 1.4723 | 33 |
| Gaia DR3 3237626100974332928 | 3237626100974332928 | 1.1285 | 9.238 | 1.1792 | 1.0058 | 5825 | 1.4423 | 33.5 |
| Gaia DR3 885545840377740544 | 885545840377740544 | 1.2036 | 9.323 | 1.2458 | 1.0091 | 5889 | 1.6817 | 32.6 |
| Gaia DR3 338349418978012672 | 338349418978012672 | 1.2896 | 9.206 | 1.1744 | 1.0089 | 5949 | 1.5563 | 29 |
| Gaia DR3 3394463531937262976 | 3394463531937262976 | 1.3702 | 9.537 | 1.2024 | 0.9198 | 5856 | 1.5318 | 39.3 |
| Gaia DR3 3153980856973522688 | 3153980856973522688 | 1.4551 | 9.509 | 1.3395 | 1.0385 | 5851 | 1.8945 | 33.8 |
| Gaia DR3 3096216879735237632 | 3096216879735237632 | 1.5228 | 9.415 | 1.1471 | 0.9366 | 5914 | 1.4502 | 34.7 |
| Gaia DR3 355877833348164352 | 355877833348164352 | 1.6180 | 9.360 | 1.1705 | 0.9970 | 5883 | 1.4785 | 32 |
| Gaia DR3 3121297732101601152 | 3121297732101601152 | 1.7011 | 9.695 | 1.3085 | 1.0594 | 5858 | 1.8165 | 31.7 |
| Gaia DR3 163797653102375424 | 163797653102375424 | 1.7786 | 9.849 | 1.2431 | 1.0289 | 5590 | 1.3594 | 41.9 |
| Gaia DR3 284804744596882944 | 284804744596882944 | 1.8133 | 9.767 | 1.3438 | 1.0515 | 5734 | 1.7587 | 36.2 |
| Gaia DR3 946341136811799680 | 946341136811799680 | 1.8729 | 9.961 | 1.1781 | 0.9319 | 5890 | 1.5049 | 36.4 |
| Gaia DR3 190627111209581696 | 190627111209581696 | 1.9757 | 10.149 | 1.2006 | 0.9974 | 5816 | 1.4859 | 35.2 |
| Gaia DR3 242194550149416832 | 242194550149416832 | 2.2312 | 10.134 | 1.1241 | 0.9123 | 5983 | 1.4587 | 32.6 |
| Gaia DR3 969515096715731328 | 969515096715731328 | 2.3069 | 10.392 | 1.1596 | 0.9822 | 5805 | 1.3757 | 35.6 |
| Gaia DR3 671574319199227648 | 671574319199227648 | 2.3622 | 10.312 | 1.1742 | 0.9106 | 5941 | 1.5475 | 35.5 |
| Gaia DR3 3367163689011899904 | 3367163689011899904 | 2.5544 | 10.598 | 1.2052 | 0.9415 | 5909 | 1.5954 | 35.5 |
| Gaia DR3 222088433846499072 | 222088433846499072 | 2.7392 | 10.621 | 1.1956 | 0.9425 | 5938 | 1.6011 | 33.9 |
| Gaia DR3 3221089377372580224 | 3221089377372580224 | 2.9038 | 10.804 | 1.1987 | 0.9489 | 5934 | 1.6051 | 33.4 |
| Gaia DR3 376368400761996288 | 376368400761996288 | 3.1045 | 10.168 | 1.1666 | 0.9422 | 5976 | 1.5638 | 31.8 |
| Gaia DR3 946284825497303936 | 946284825497303936 | 3.1235 | 11.149 | 1.1550 | 0.9488 | 5833 | 1.3913 | 36.8 |
| Gaia DR3 3157053128680942464 | 3157053128680942464 | 3.1758 | 11.119 | 1.1765 | 0.9937 | 5588 | 1.2159 | 44.4 |
| Gaia DR3 339107016849361280 | 339107016849361280 | 3.4134 | 10.901 | 1.1790 | 0.9100 | 5991 | 1.6133 | 33.5 |
| Gaia DR3 3151845021274196864 | 3151845021274196864 | 3.5360 | 11.286 | 1.3080 | 0.9992 | 5952 | 1.9345 | 31.5 |
| Gaia DR3 3163337249950832512 | 3163337249950832512 | 3.6746 | 11.512 | 1.3133 | 1.0461 | 5805 | 1.7645 | 34.2 |
| Gaia DR3 389967774754460160 | 389967774754460160 | 3.8843 | 10.603 | 1.2553 | 1.0043 | 5991 | 1.8289 | 29.3 |

## Rebuild

Run `python rebuild_configs.py` to rebuild all catalogued stars from the shared template, or add `--check` to verify existing files. New configs are directly in `RealGalacticSystems/Systems/MilkyWay/Systems`. They keep referenceBody=Sun, period=3E300, mode=OFF and recoveryValue rounded up. The earlier 90 catalog records/configs, template, Sun, galaxy definitions and README are preserved. See [StarGeneration.md](StarGeneration.md) for the workflow.
