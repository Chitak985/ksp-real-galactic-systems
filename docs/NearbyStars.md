# Nearby single Sun-like stars

This document describes the first nine additions to Tau Ceti, researched on
2026-10-10. [AdditionalNearbyStars.md](AdditionalNearbyStars.md) documents the
next 30 additions, which brought the catalog to 40 stars.
[OuterDiskStars.md](OuterDiskStars.md) adds 50 distant candidates for a total of 90.
The first additions are
ordered by distance from Sol and use the shared Sun-like
template. Every entry has a sourced radius, mass, effective temperature,
bolometric luminosity, rotation period, and ICRS J2000 position.

For this batch, Sun-like means a G-type dwarf or near-main-sequence G star in
the adopted physical reference. Established subgiants and confirmed stellar
multiple systems are excluded. This is a selection of nearby G stars with
complete adopted data, rather than a census of all nearby F/G/K solar analogues.
Beta Comae Berenices lies at the F9.5/G0 classification boundary and is included
using the G0V classification in the adopted study. Iota Persei and Lambda Aurigae
have mixed V/IV-V classifications; their published larger radii are retained.

## Added stars, in distance order

Radius, mass, and luminosity below are in solar units. Distances are calculated
from [SIMBAD](https://simbad.cds.unistra.fr/simbad/sim-tap) Gaia EDR3 parallaxes;
the exact coordinates, parallaxes, uncertainties, and reference bibcodes are
saved in `data/bodies.json`. Table values are rounded for readability.

| Star | HD | Distance (ly) | Radius | Mass | Temperature (K) | Luminosity | Rotation (days) | Data references |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 82 G Eridani | 20794 | 19.704 | 0.930 | 0.790 | 5368 | 0.6869 | 35.0 | N25 |
| Beta Canum Venaticorum | 109358 | 27.634 | 1.123 | 0.852 | 5653 | 1.1510 | 16 (estimated) | B12, W04 |
| 61 Virginis | 115617 | 27.836 | 0.9867 | 0.930 | 5538 | 0.8222 | 32.1 | V14, Y24 |
| Beta Comae Berenices | 114710 | 29.998 | 1.106 | 1.045 | 5936 | 1.3570 | 12.35 | B12, M18 |
| Kappa1 Ceti | 20630 | 30.255 | 0.919 | 1.037 | 5776 | 0.8410 | 9.15 | B12, S20 |
| 61 Ursae Majoris | 101501 | 31.233 | 0.860 | 0.900 | 5502 | 0.6166 | 16.28 | C21 |
| Iota Persei | 19373 | 34.499 | 1.412 | 1.169 | 5915 | 2.1810 | 19 (estimated) | B12, W04 |
| Lambda Serpentis | 141004 | 38.865 | 1.363 | 1.090 | 5901 | 1.9840 | 24.3 | M23 |
| Lambda Aurigae | 34411 | 40.973 | 1.331 | 1.041 | 5749 | 1.7320 | 27.9 (estimated) | B12, S20 |

The saved sources distinguish observed periodic modulation from empirical
rotation estimates. Beta Canum Venaticorum and Iota Persei use the activity-based
estimates published by Wright et al.; Lambda Aurigae uses Schmitt and Mittag's
calculated period, whose observed-period column is unavailable. These are
published estimates, not substituted solar defaults or periods calculated from
`v sin i`. Their `rotation_period_kind` is `activity_calibration`.

The other six stars use periods detected in photometry or spectroscopic activity
time series (`observed_modulation`). For 82 G Eridani, the adopted FWHM result
is 35.0 +3.2/-2.5 days; the same paper finds 38.8 +2.4/-2.6 days from BIS.
For 61 Ursae Majoris, 16.28 days is the published photometric GP maximum-a-posteriori
value. Stellar differential rotation and changing active regions can produce
different period determinations; the catalog records the adopted method.

## Why nearer candidates were skipped

The G-star search covered SIMBAD entries with parallax above 65 mas, then checked
companion status and physical data. The following nearer candidates were not
added. Distances are approximate and are used only to explain selection order.
K/M dwarfs and other spectral classes await their own templates.

| Candidate | Approximate distance (ly) | Reason |
| --- | ---: | --- |
| Alpha Centauri A | 4.4 | Confirmed multiple system |
| Tau Ceti | 11.9 | Already in the catalog |
| Eta Cassiopeiae A | 19.4 | Confirmed binary |
| Delta Pavonis | 19.9 | G8IV classification; excluded from this dwarf batch |
| Xi Bootis A | 22.0 | Confirmed binary |
| Beta Hydri | 24.3 | Established subgiant |
| Mu Cassiopeiae A | 25.0 | Confirmed binary |
| Mu Herculis | 27.2 | Confirmed multiple system and evolved primary |
| Zeta Tucanae | 28.1 | Current F9.5V classification; outside the adopted G-star batch |
| Chi1 Orionis | 28.3 | Confirmed binary |
| Xi Ursae Majoris B | 28.5 | Confirmed multiple system |
| 41 Arae | 28.7 | Confirmed multiple system |
| Gamma Pavonis | 30.2 | Current F9V classification; outside the adopted G-star batch |
| HD 102365 | 30.4 | Confirmed stellar companion |
| Alpha Mensae | 33.3 | Confirmed stellar companion |
| Zeta Herculis | 35.0 | Confirmed binary and evolved primary |
| Delta Trianguli | 35.6 | Confirmed binary |
| 11 Leonis Minoris | 36.6 | Confirmed stellar companion |
| Eta Bootis | 37.2 | Confirmed binary and evolved primary |
| Zeta1 and Zeta2 Reticuli | 39.3 | Wide stellar pair |
| 85 Pegasi | 40.2 | Confirmed binary |
| 44 Bootis | 40.8 | Confirmed multiple system |
| UCAC4 767-032810 | 40.9 | Bare G classification; no complete adopted physical/rotation package |

Multiplicity decisions use [Raghavan et al. (2010), Table 13 and its notes](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/190/1?format=html&tex=true),
with [SIMBAD](https://simbad.cds.unistra.fr/simbad/sim-id) classifications as a
cross-check. Beta Canum Venaticorum is included because the survey explicitly
rejects the historical spectroscopic/speckle binary claims and identifies its
wide apparent companion as optical. A generic double-star tag alone is not a
confirmed bound companion. Beta Hydri's evolved status is independently
established by [Metcalfe et al. (2024)](https://arxiv.org/abs/2408.05482).
"Single" records the assessment of the consulted literature; future companion
discoveries may require revising an entry.

## Physical and rotation references

- **N25:** [Nari et al. (2025)](https://arxiv.org/html/2501.17092v1), Table 1 and activity analysis: 82 G Eridani.
- **B12:** [Boyajian et al. (2012), VizieR catalog](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJ/746/101?format=html&tex=true), Tables 10 and 12: the five stars marked B12. Radii, luminosities, and temperatures are paired with the same study's isochrone masses. The Beta Canum Venaticorum package is also adopted by [Kane et al. (2020), Sections 3.1 and 3.2](https://arxiv.org/pdf/2006.13962).
- **V14:** [von Braun et al. (2014)](https://academic.oup.com/mnras/article/438/3/2413/972333), Table 4: 61 Virginis physical properties.
- **W04:** [Wright et al. (2004), VizieR catalog](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/152/261?format=html&tex=true), Table 2: Beta Canum Venaticorum and Iota Persei activity-calibrated rotation estimates.
- **Y24:** [Yu et al. (2024)](https://pure-oai.bham.ac.uk/ws/files/217941536/YuH2024Modelling_AAMonline.pdf), Tables 1 and B1: 61 Virginis observed rotation.
- **M18:** [Mittag et al. (2018)](https://arxiv.org/pdf/1807.05825), Table A.1, HD 114710: Beta Comae Berenices rotation, attributed to Wright et al. (2011).
- **S20:** [Schmitt and Mittag (2020)](https://onlinelibrary.wiley.com/doi/full/10.1002/asna.202013790), Table 4: observed Kappa1 Ceti rotation and calculated Lambda Aurigae rotation.
- **C21:** [Cabot et al. (2021)](https://arxiv.org/html/2010.14717v1), Table 1, Table 3 and Section IV.1: 61 Ursae Majoris. The reported log luminosity of -0.21 is converted with `10**(-0.21)`.
- **M23:** [Metcalfe et al. (2023)](https://arxiv.org/pdf/2308.09808), Table 5 and Section 2.4: Lambda Serpentis.

Conversions use 695,700,000 m per solar radius and 1.98847e30 kg per solar mass,
matching the existing Tau Ceti entry. Physical values are adopted from the cited
studies without refitting them to the newer placement parallaxes. Published
uncertainties are saved where provided; missing uncertainties are documented
rather than invented. The common rendering color comes from the supplied
Sun-like template and is an aesthetic setting.

All catalogued stars use `referenceBody = Sun`, `period = 3E300`, and
`mode = OFF`. Their hidden orbits encode heliocentric positions in the existing
Galactic coordinate adapter. Run `python rebuild_configs.py` after editing the
catalog or shared template. See [StarGeneration.md](StarGeneration.md) for the
rebuild and verification commands.
