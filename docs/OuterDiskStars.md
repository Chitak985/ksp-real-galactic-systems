# Outer-disk Sun-like candidates

This batch added **50 real catalogued stars**, bringing the saved catalog to **90** bodies at the time. [DiskGapStars.md](DiskGapStars.md) documents 60 later additions between the nearby and outer-disk samples, bringing the current catalog to 150. These entries use single-star configurations for candidates with no known stellar companion, model estimates and saved uncertainty ranges. Each game description briefly states the uncertainty; the objects are not certified isolated stars. Researched on 2026-10-14.

Each new configuration contains this description (fewer than 100 characters):

> Outer-disk Sun-like candidate; distance, properties and spin estimated; single status unconfirmed.

## Where these stars are

The selected distance medians place the candidates about **4.12-7.85 kpc (13,437-25,619 light-years) from Sol**, at cylindrical Galactic radii about **12.26-15.28 kpc (39,975-49,848 light-years)** for an adopted Solar radius R0=8.2 kpc. These are the Milky Way's outer disk and the vicinity/beyond its star-forming disk break, not a claim about the outermost halo or a sharp edge to the entire Galaxy. [Fiteni et al. (2026)](https://arxiv.org/abs/2603.18737) place the star-forming disk break around 11.28-12.15 kpc; older populations extend beyond it.

For each source, transform the published StarHorse 5th-95th distance interval through R=sqrt((R0-d*cos(b)*cos(l))**2+(d*cos(b)*sin(l))**2). Every adopted median is at R>=12 kpc and the minimum R over that interval is >=11.5 kpc. The interval is 90% of the distance model's posterior; it does not certify a 90% probability of singleness, every adopted point value's accuracy, disk membership or the model itself. The median |z| is below 2 kpc; these are spatial outer-disk candidates without a new kinematic membership determination.

## Selection and screening

The search used the corrected StarHorse SDSS DR12 catalog over all Galactic longitudes and |b|<25 degrees, plus an anticenter LAMOST LRS search. Cuts: estimated Teff 5300-6000 K; logg 4.2-4.65 with the 16th percentile at least 4.0; mass 0.7-1.25 solar masses; derived radius 0.7-1.5 solar radii; no StarHorse output or input/output age-consistency warning; a published MSTO-regime age for spin modeling; and a 5th-95th distance interval narrower than 70% of its median. MSTO denotes main-sequence turnoff; these entries retain dwarf-sized radii and exclude the separate subgiant class. All 50 final sources are from SDSS DR12.

69 candidates survived the physical/location checks. Companion/astrometric screening excluded 13: nine had multiple image-peak detections, two had a >3-sigma parallax/distance inconsistency (with a 0.03-mas systematic term), and two had published F9 classifications rather than the adopted G-like type. All 50 selected stars have Gaia NSS=0, RUWE<1.2, duplicated_source=false and ipd_frac_multi_peak=0. There are no entries for these candidates in Gaia's variability classifier or rotation-modulation table; the SIMBAD identifiers that matched have no stellar binary flag. Missing records do not constitute a companion search. The first 50 of the remaining 56 were chosen in descending estimated Galactic radius.

The metadata `multiplicity_status=no_known_stellar_companion` and `data_status=estimated_candidate` distinguish them from the earlier nearby-star records. `system_type=single` is the supported configuration model. Gaia DSC star/binary probabilities are not used, as [the documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_dsc.html) warns against using those probabilities to build binary samples.

## Physical and spin estimates

Mass, temperature, gravity, metallicity, distance and age use the posterior medians in [Queiroz et al. (2023), corrected StarHorse SDSS DR12](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A+A/673/A155?format=html&tex=true). The published percentiles, identifiers and quality flags are saved for every star. These are single-star evolutionary-model estimates constrained by spectroscopy, photometry and Gaia astrometry. Distances are not computed as 1/parallax, because these faint sources' parallaxes are noisy and sometimes negative.

Radius is derived with R=sqrt(G*M/g), with g=10**logg cm/s² converted to SI. Luminosity follows L/Lsun=(R/Rsun)**2*(T/5772 K)**4. Constants match the earlier catalog (695,700,000 m per solar radius and 1.98847e30 kg per solar mass), with G=6.67430e-11 SI. Adopted temperature is rounded to 1 K, radius/luminosity to 0.0001 solar units. The JSON separates `reported_values` from `derived_values`. Radius/luminosity sensitivity ranges use the corners of the published marginal 16th/84th percentile inputs; they are not joint credible intervals because parameter correlations are unavailable.

Every new spin value is an unmeasured `gyrochronology_estimate`: [Barnes (2007)](https://arxiv.org/abs/0704.3068) P_days=0.7725*(B-V-0.4)**0.601*(age_Myr)**0.5189. Intrinsic B-V is inferred by inverting [Ballesteros (2012), Equation 14](https://arxiv.org/abs/1201.1809), using the model temperature. The adopted period is rounded to 0.1 day. Age 5th/95th and temperature 84th/16th percentiles give an input-sensitivity range, not a validated prediction interval. No rotation measurement or color observation is claimed.

These are old stars, so the classical spin relation can be systematically wrong: [van Saders et al. (2016)](https://arxiv.org/abs/1601.02631) show weakened magnetic braking and faster rotation than simple gyrochronology predicts. Metallicity, differential rotation and an unseen companion also add uncertainty. The estimates are for the approved game model; **a 90% accuracy claim is not supported for rotation**. This limitation is recorded with every period and summarized as estimated spin in each game description.

## Added stars

Units: distances in kiloparsecs, radius/mass/luminosity in solar units, temperature in kelvin and spin in days. All rows are estimates. Gaia IDs are saved as strings to prevent 64-bit identifiers being rounded by JavaScript-based readers.

| Gaia DR3 source ID | Sol distance | Galactic radius | Radius | Mass | Temperature | Luminosity | Spin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 238900477737876352 | 7.855 | 15.284 | 1.1132 | 0.9415 | 5980 | 1.4277 | 31.6 |
| 3161317996847015424 | 6.332 | 14.188 | 1.0912 | 0.8940 | 5937 | 1.3328 | 35.2 |
| 3386405172561319168 | 6.094 | 14.137 | 1.0814 | 0.9388 | 5930 | 1.3028 | 32.6 |
| 3353501325026684800 | 6.018 | 13.901 | 1.0798 | 0.9285 | 5969 | 1.3335 | 31.3 |
| 3394367148574000384 | 5.811 | 13.869 | 1.0844 | 0.8726 | 5969 | 1.3449 | 34.6 |
| 3161364519931510272 | 5.952 | 13.818 | 0.9894 | 0.8947 | 5849 | 1.0322 | 35.9 |
| 3145671190467671936 | 6.275 | 13.631 | 0.9711 | 0.8383 | 5902 | 1.0309 | 37.6 |
| 3384413716486072960 | 5.469 | 13.591 | 1.0640 | 0.8905 | 5968 | 1.2939 | 33.6 |
| 886320892995828608 | 5.647 | 13.546 | 1.0876 | 0.8924 | 5918 | 1.3072 | 36.4 |
| 3160891523774758656 | 5.442 | 13.328 | 1.1184 | 0.9329 | 5929 | 1.3926 | 33.7 |
| 888901378067020800 | 5.292 | 13.312 | 1.1295 | 0.9370 | 5880 | 1.3740 | 34.3 |
| 887938072738494208 | 5.303 | 13.283 | 1.0106 | 0.8726 | 5921 | 1.1309 | 36.0 |
| 398388526290739712 | 6.827 | 13.242 | 1.1482 | 0.9605 | 5865 | 1.4054 | 33.8 |
| 3394014205344100736 | 5.107 | 13.162 | 1.0490 | 0.9578 | 5819 | 1.1367 | 33.6 |
| 888405154726660352 | 5.061 | 13.111 | 1.0327 | 0.9207 | 5771 | 1.0657 | 38.7 |
| 3432546555979369600 | 4.861 | 12.999 | 0.9844 | 0.9152 | 5812 | 0.9962 | 34.4 |
| 888179617403534464 | 4.977 | 12.994 | 1.0112 | 0.8512 | 5980 | 1.1781 | 34.8 |
| 237937855304935296 | 5.294 | 12.962 | 1.0767 | 0.9016 | 5933 | 1.2941 | 34.8 |
| 283518144196539008 | 5.354 | 12.944 | 0.9364 | 0.8492 | 5803 | 0.8958 | 39.6 |
| 3160666952821149568 | 5.039 | 12.930 | 0.9608 | 0.8696 | 5871 | 0.9881 | 36.3 |
| 3160795075989674624 | 4.993 | 12.909 | 0.9954 | 0.8613 | 5955 | 1.1226 | 34.6 |
| 3384425609251487744 | 4.734 | 12.863 | 0.9654 | 0.8873 | 5864 | 0.9929 | 35.1 |
| 3384154506619678208 | 4.712 | 12.828 | 1.1878 | 0.9206 | 5954 | 1.5974 | 34.0 |
| 887737716807748864 | 4.787 | 12.826 | 0.9599 | 0.8764 | 5866 | 0.9829 | 36.6 |
| 948852352650054912 | 5.008 | 12.813 | 0.9157 | 0.8307 | 5805 | 0.8578 | 40.2 |
| 3393707235438955392 | 4.734 | 12.794 | 1.1156 | 0.9495 | 5893 | 1.3523 | 34.2 |
| 400846583316486528 | 6.295 | 12.720 | 1.0535 | 0.8934 | 5819 | 1.1465 | 39.2 |
| 3384299302853236864 | 4.577 | 12.715 | 0.9811 | 0.9253 | 5771 | 0.9619 | 35.7 |
| 400840437221584512 | 6.277 | 12.709 | 1.1209 | 0.9039 | 5928 | 1.3979 | 35.2 |
| 948535968179617408 | 4.809 | 12.645 | 0.9531 | 0.8361 | 5894 | 0.9877 | 37.7 |
| 3146942741306828928 | 5.155 | 12.577 | 1.0520 | 0.8860 | 5850 | 1.1677 | 38.6 |
| 398392473363742720 | 5.977 | 12.541 | 0.9961 | 0.8571 | 5936 | 1.1099 | 36.0 |
| 137119270466717440 | 5.061 | 12.513 | 1.0446 | 0.8755 | 5870 | 1.1672 | 38.2 |
| 668431051676489216 | 4.856 | 12.455 | 1.0617 | 0.8547 | 5978 | 1.2969 | 35.2 |
| 893055165494221696 | 4.553 | 12.437 | 0.9263 | 0.8447 | 5793 | 0.8706 | 40.0 |
| 3353082101858810112 | 4.476 | 12.417 | 1.0291 | 0.8734 | 5940 | 1.1878 | 35.0 |
| 3160541260602855680 | 4.494 | 12.414 | 0.9877 | 0.9356 | 5909 | 1.0715 | 28.9 |
| 3161332702813382912 | 4.470 | 12.401 | 0.9397 | 0.8754 | 5821 | 0.9134 | 36.1 |
| 3353126838240287360 | 4.451 | 12.398 | 1.0767 | 0.9235 | 5857 | 1.2291 | 35.9 |
| 3383985628504131328 | 4.204 | 12.344 | 1.0470 | 0.9382 | 5937 | 1.2270 | 29.4 |
| 889481168588600704 | 4.340 | 12.343 | 0.9659 | 0.8781 | 5803 | 0.9532 | 38.1 |
| 3384356271299775616 | 4.206 | 12.338 | 1.0597 | 0.9407 | 5980 | 1.2938 | 30.2 |
| 894285068033478272 | 4.519 | 12.306 | 1.1463 | 0.9111 | 5998 | 1.5322 | 32.4 |
| 136301062016708992 | 4.872 | 12.301 | 0.9329 | 0.8363 | 5760 | 0.8631 | 41.4 |
| 875774617980902528 | 4.569 | 12.297 | 0.9783 | 0.8308 | 5970 | 1.0953 | 35.5 |
| 3160930354575369344 | 4.344 | 12.281 | 1.0068 | 0.9252 | 5844 | 1.0652 | 33.0 |
| 3168721219090912256 | 4.461 | 12.274 | 0.9368 | 0.8944 | 5597 | 0.7759 | 43.0 |
| 3394848287991539072 | 4.185 | 12.269 | 1.0395 | 0.8866 | 5970 | 1.2366 | 32.5 |
| 3384245181967956864 | 4.120 | 12.266 | 1.0670 | 0.9420 | 5987 | 1.3178 | 29.5 |
| 875367695599486720 | 4.506 | 12.256 | 0.9808 | 0.8321 | 5930 | 1.0717 | 37.1 |

## Rebuilding and preservation

Run `python rebuild_configs.py` to rebuild all catalogued stars. `python rebuild_configs.py --check` verifies that the saved configurations match the data and template. Files in this batch are directly under `RealGalacticSystems/Systems/MilkyWay/Systems`, with names `gaia_dr3_[source_id].cfg`. Positions use the existing Galactic-axis adapter and frozen Sun-parented orbits (period=3E300, mode=OFF); recoveryValue is rounded up.

Gaia directions are propagated from J2016 to J2000 using tangent-plane proper motions before conversion. Perspective acceleration is neglected; the angular correction is tiny relative to the distance uncertainty. The 8.2-kpc Solar Galactocentric radius is used only to document/select the physical region and does not modify the existing SagittariusA orbit or GalacticDisk. The first 40 catalog records, shared template and earlier config contents are preserved; README is untouched. Existing curves are retained.

See [StarGeneration.md](StarGeneration.md) for the compiler workflow and the two nearby-star documents for the earlier additions.
