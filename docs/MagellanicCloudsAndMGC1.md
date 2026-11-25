# Magellanic Clouds and MGC1 visuals

Each existing body has one appended `GalacticDiskDefinition`. The original body node is preserved except for a description that explains the visual approximation. Positions, masses, SOIs, root structure, cache paths and existing flags remain unchanged. The manual orbit updater preserves these appended definitions. These are bespoke manual visuals; no irregular-galaxy family is added to the bulk importer.

## Large Magellanic Cloud

The volumetric model has a dominant stellar bar, blue diffuse disk, one weak broad arm, and scattered dust and star-forming regions. The representative plane uses inclination 25.86 degrees and line-of-nodes PA 149.23 degrees from [Choi et al. (2018)](https://arxiv.org/abs/1804.07765). The existing barycenter is retained; their fitted center differs from this body position.

The 5-kpc display radius, bar sky PA 115 degrees, bar dimensions, arm phase, clumps and exposure are visual estimates. The renderer's bar is centered, so the observed offset, warp, outer low-surface-brightness populations and full disturbed morphology are not reproduced. Its 18,000 points sample stellar light rather than count individual real stars.

## Small Magellanic Cloud

The volumetric definition approximates an elongated, deep stellar cloud with a bar-like concentration, scattered dust and faint luminous knots. It has no spiral arms. A 45-degree bar sky angle is consistent with the approximate orientation discussed by [Lobo Gomes et al. (2015)](https://arxiv.org/abs/1504.02104). [Cepheid morphology](https://doi.org/10.1093/mnras/stz2328) and [the VMC expansion analysis](https://doi.org/10.1051/0004-6361/202659431) show why a single thin rotating disk is inadequate.

The 3-kpc display radius, sky axis ratio 0.65, line-of-sight Gaussian sigma 3 kpc, colors and clumps are estimates. Local Y is placed along the sightline: the registry's zero inclination describes this modeling convention, not a measured face-on disk or dynamical axis. Different populations trace different structures. The 12,000 points are representative samples.

## MGC1

`renderMode = Particles` creates 25,000 warm-white sample stars in a spherical Gaussian concentration, with no disk, spiral arms, gas or diffuse volume. This is the isolated M31 globular cluster MGC1, not the unrelated star MGC 1 or Mackey-GC1. [Mackey et al. (2010), section 4 and Table 3](https://academic.oup.com/mnras/article/401/1/533/1008146) give a half-light radius of approximately 7.5 pc (2.5 arcsec) and identify an old, metal-poor population.

For a projected Gaussian, the half-light radius is `sqrt(2 ln 2) * sigma`. With all three `bulgeScale` values set to `100/3.5`, the physical scale is chosen so this radius is 7.5 pc. The Gaussian approximates the visible concentration; it does not fit a King/EFF profile or reproduce the extremely faint extended halo. Sprite sizes and colors are display estimates. The particle count is a rendering budget, not a physical star count. The stock particle branch ignores volumetric brightness parameters, so its appearance is controlled through particle size, count and RGBA colors.

## Shared geometry and visibility

Scales use meters per parsec and the existing scaled-space factor of 6000:

```text
objectScale = display_radius_pc * PARSEC_M / (100 * 6000)
```

Fixed Euler rotations use the same ICRS-to-Galactic-to-Unity adapter as the other objects. `alignRotationToOrbit = false` and `rotationPeriod = 0` keep the visuals static. The manual orientation registry records the adopted measurements, sources, estimates and complete visual parameter sets for these three objects. MGC1 is explicitly identified as a globular cluster there.

The cloud interiors use opacity multiplier 0.2; MGC1 uses 1 to keep its stars visible from within the cluster. All three use daylight-atmosphere multiplier 0.01 and night-atmosphere multiplier 0.65. These are display settings. The apparent size of MGC1 is tiny from Sol; approaching or zooming is needed to resolve its stars.

## Verification

The final configurations were rendered with the current GalacticDisk source and packaged shaders in an isolated Unity project. The previews use an enlarged, normalized-scale camera and substitute KSP APIs; they are not a live-game test. MGC1 generated all 25,000 particles, passed the spherical-distribution check, and its sampled projected half-radius was within 0.43 percent of the Gaussian expectation. Independent checks verify the physical scale, sky orientation, LMC bar angle and preservation of existing body content.

The legacy-host orbit test now checks the absolute 100-pc core cap and that the SOI excludes Sol. Its former 0.1-percent-of-distance threshold rejected the LMC's unchanged 100-pc SOI at its nearby distance; no physical setting was changed to accommodate the test. The other 77 available regression cases passed, and all four orientation/legacy-host tests passed after this correction. The retired fictional Unknown object is excluded from the active manual registry and validation; its old configuration is not required or recreated.
