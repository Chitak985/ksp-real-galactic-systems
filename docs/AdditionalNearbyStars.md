# Additional nearby single Sun-like stars

The second batch adds **30 stars**, bringing `data/bodies.json` to **40 generated stars** including Tau Ceti. Researched on 2026-10-10, it extends the nearby-star search to **25 parsecs (81.54 light-years)** and orders the additions by distance from Sol.

Every addition has a published radius, mass, effective temperature, bolometric luminosity, rotation value and ICRS J2000 position. Eight use observed rotation periods; 22 use explicitly labeled published activity-calibrated estimates. The existing ten catalog entries and their config contents are preserved.

## Selection

The search crossmatches nearby SIMBAD G-type entries with Brewer et al.'s physical catalog, the Raghavan et al. multiplicity survey and published rotation studies. Eligible entries have dwarf or mixed IV-V classifications, complete physical parameters and spectroscopic log(g) at least 4.1. Confirmed stellar companions, unresolved double/binary-related flags, established giants/subgiants and missing required values are skipped. This is a bounded, reviewed selection, not a complete census of every eligible star in the Milky Way.

The adopted records report no stellar companions for the selected stars. For 47 Ursae Majoris, the companion survey lists planets b and c, which do not make it a stellar binary. BY Draconis or eruptive variability alone is not used as a binary exclusion. As in the first batch, mixed IV-V classifications are allowed: 58 Eridani, HD 196761 and HD 78366 have dwarf-sized published radii. 15 Leonis Minoris also has G0 IV-V classifications in the literature; its larger 1.56-solar-radius measurement is retained. HD 124292 lies on the G8/K0V boundary and has an adopted temperature of 5458 K.

## Added stars

Radius, mass and luminosity are in solar units. All physical rows use **B16**. Positions and distances use the SIMBAD Gaia EDR3 values saved in the JSON; table values below are rounded for display.

| Star | HD | Distance (ly) | Radius | Mass | Temperature (K) | Luminosity | Rotation (days) | Rotation reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HD 14412 | 14412 | 41.861 | 0.76 | 0.77 | 5373 | 0.4365 | 29 | W04 estimated |
| 58 Eridani | 30495 | 43.183 | 0.95 | 1.04 | 5834 | 0.9333 | 11.1 | S20 observed |
| 47 Ursae Majoris | 95128 | 45.295 | 1.22 | 1.09 | 5872 | 1.6218 | 26.5 | S20 estimated |
| Pi1 Ursae Majoris | 72905 | 47.093 | 0.95 | 1.04 | 5866 | 0.9550 | 5.22 | M18 |
| HD 196761 | 196761 | 47.855 | 0.81 | 0.86 | 5473 | 0.5248 | 31 | W04 estimated |
| HD 38858 | 38858 | 49.610 | 0.91 | 0.95 | 5735 | 0.8128 | 24.4 | S20 estimated |
| HD 152391 | 152391 | 54.834 | 0.86 | 0.91 | 5425 | 0.5888 | 11.43 | M18 |
| HD 111395 | 111395 | 55.767 | 0.92 | 1.00 | 5617 | 0.7586 | 16.76 | M24 |
| 37 Geminorum | 50692 | 56.766 | 1.07 | 1.02 | 5913 | 1.2589 | 15 | W04 estimated |
| 15 Leonis Minoris | 84737 | 61.391 | 1.56 | 1.19 | 5872 | 2.5704 | 40.5 | S20 observed |
| HD 52711 | 52711 | 61.806 | 1.10 | 1.03 | 5886 | 1.3183 | 18 | W04 estimated |
| HD 78366 | 78366 | 61.806 | 1.04 | 1.11 | 5984 | 1.2589 | 9.67 | M18 |
| HD 9407 | 9407 | 68.649 | 0.99 | 1.00 | 5672 | 0.9120 | 30 | W04 estimated |
| 51 Arietis | 18803 | 69.129 | 0.95 | 1.02 | 5648 | 0.8318 | 27 | W04 estimated |
| HD 124292 | 124292 | 71.065 | 0.91 | 0.88 | 5458 | 0.6607 | 34 | W04 estimated |
| HD 45184 | 45184 | 71.403 | 1.03 | 1.07 | 5854 | 1.1220 | 19.98 | F16 |
| HD 90156 | 90156 | 71.584 | 0.90 | 0.91 | 5635 | 0.7413 | 25 | W04 estimated |
| HD 97343 | 97343 | 72.082 | 0.94 | 0.88 | 5387 | 0.6761 | 37 | W04 estimated |
| HD 98281 | 98281 | 72.190 | 0.82 | 0.84 | 5381 | 0.5129 | 33 | W04 estimated |
| HD 4915 | 4915 | 73.144 | 0.84 | 0.93 | 5668 | 0.6457 | 23 | W04 estimated |
| HD 168009 | 168009 | 75.966 | 1.14 | 1.05 | 5808 | 1.3183 | 26 | W04 estimated |
| HD 127334 | 127334 | 76.289 | 1.21 | 1.07 | 5671 | 1.3804 | 33 | W04 estimated |
| HD 1461 | 1461 | 76.328 | 1.10 | 1.07 | 5739 | 1.1749 | 29 | W04 estimated |
| HD 12846 | 12846 | 77.446 | 0.91 | 0.93 | 5700 | 0.7943 | 29.6 | S20 observed |
| HD 92719 | 92719 | 77.826 | 0.95 | 1.00 | 5796 | 0.9333 | 20.9 | S20 estimated |
| HD 176377 | 176377 | 78.188 | 0.92 | 0.99 | 5877 | 0.9120 | 17 | W04 estimated |
| HD 20619 | 20619 | 78.269 | 0.92 | 0.95 | 5732 | 0.8318 | 21 | W04 estimated |
| HD 159222 | 159222 | 78.938 | 1.05 | 1.11 | 5870 | 1.1749 | 22 | W04 estimated |
| HD 42618 | 42618 | 79.596 | 0.93 | 0.99 | 5755 | 0.8511 | 26 | S20 estimated |
| HD 51419 | 51419 | 80.986 | 0.95 | 0.90 | 5732 | 0.8710 | 21.4 | S20 estimated |

## Adopted values and limitations

B16 supplies the physical parameters together; masses use its isochrone `Miso` column, rather than the noisier mass calculated from spectroscopic gravity and radius. Luminosities are converted from `logL` with `10**logL`. Published radius errors, logarithmic luminosity errors and mass interval bounds are saved with their units. The 25 K temperature precision describes the study overall and is not represented as an individual error bar. The corrected electronic catalog associated with the 2017 erratum is used.

A rotation value tagged `activity_calibration` is a published empirical estimate. It is not an observed period or a substituted solar default. W04's blank calibration flags were checked; its integer periods have no individual uncertainty in that table. S20 calculated periods retain their published errors. Where available, measured periodic modulation is preferred. The source records preserve methods, quality flags and uncertainties without inventing missing error bars.

HD 38858 and 47 Ursae Majoris use the S20 calculated periods because the observed summaries have quality 3 and require confirmation. 15 Leonis Minoris uses the authors' quality-2 observed summary of 40.5 days; confirmation remains desirable. Pi1 Ursae Majoris uses the directly measured 5.22-day M18 compilation value, consistent with the 5.23 +/- 0.02-day TIGRE season in S20. HD 111395 uses 16.76 +/- 0.36 days from the M24 abstract, analysis and Table 7; the conclusion prints 16.36 days. Its seasonal changes are interpreted as differential rotation.

Examples skipped pending further review include 18 Scorpii, which has a current SIMBAD double-star flag despite the older survey's single entry, and 72 Herculis, whose red-giant object-type flag conflicts with its G0V classification and dwarf-sized physical parameters. These flags are not treated as proof that those stars are physical binaries or giants. HD 42807's RS CVn classification also needs resolution before selection. Psi Serpentis (HD 140538) has a confirmed stellar companion in the survey and is excluded. HD 71148 and HD 76151 do not have the complete adopted B16 physical package used for this batch.

## References

- **B16**: [Brewer et al. (2016), ApJS 225, 32, corrected tables 8 and 11](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/225/32?format=html&tex=true); [paper](https://arxiv.org/abs/1606.07929).
- **W04**: [Wright et al. (2004), ApJS 152, 261, Table 2](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/152/261?format=html&tex=true); activity-calibrated rotation estimates.
- **S20**: [Schmitt and Mittag (2020), Astronomische Nachrichten 341, 497, Tables 2 and 4 and Section 10](https://onlinelibrary.wiley.com/doi/full/10.1002/asna.202013790); observed and calculated columns kept separate.
- **M18**: [Mittag, Schmitt and Schroeder (2018), A&A 618, A48, Section 2 and Table A.1](https://arxiv.org/pdf/1807.05825); directly measured period compilation.
- **F16**: [Flores et al. (2016), A&A 589, A135, Section 3.3](https://arxiv.org/pdf/1604.01307); HD 45184's rotation.
- **M24**: [Mittag et al. (2024), A&A 682, A86](https://doi.org/10.1051/0004-6361/202345919); HD 111395's rotation and differential rotation.
- **Multiplicity**: [Raghavan et al. (2010), ApJS 190, 1, Table 13](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/190/1?format=html&tex=true); checked alongside current SIMBAD object-type records. Table 13 includes planetary entries as well as stellar companions.
- **Placement**: [SIMBAD](https://simbad.cds.unistra.fr/simbad/sim-tap); per-star query links, coordinate values, parallaxes, errors, classifications and bibcodes are saved in the catalog.
- **15 Leonis Minoris classification**: [Soubiran et al. (2020), A&A 637, A43](https://www.aanda.org/articles/aa/pdf/2020/05/aa37216-19.pdf), HD 84737 G0 IV-V.

## Rebuild and placement

Run `python rebuild_configs.py` to rebuild all 40 generated stars from the shared template. `python rebuild_configs.py --check` verifies that the saved files match the catalog and template. All generated files are directly inside `RealGalacticSystems/Systems/MilkyWay/Systems`. They use hidden, frozen Sun-parented positioning orbits (`period = 3E300`, `mode = OFF`) and integer recovery values rounded up. No galaxies, binaries, planets or stock-Sun changes are added by this batch.

See [StarGeneration.md](StarGeneration.md) for the rebuild workflow and [NearbyStars.md](NearbyStars.md) for the first nine additions.
