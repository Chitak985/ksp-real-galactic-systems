# Milky Way disk appearance

The two top-level `GalacticDiskDefinition` nodes in
`RealGalacticSystems/Systems/MilkyWay/Configs/SagittariusA.cfg` are visual layers
attached to the existing SagittariusA body. They do not add celestial bodies or
change stellar positions, orbits, SOIs, or the stock Sun.

## Layers

| Setting | Bright thin disk and bulge | Diffuse thick disk |
| --- | ---: | ---: |
| Particles (`starCount`) | 240,000 | 60,000 |
| `diskThickness` | 5 | 15 |
| Vertical Gaussian standard deviation, generator units | 1.1 | 3.3 |
| Bulge fraction | 0.275 | 0 |
| Arm / interarm alpha | 0.85 / 0.7 | 0.25 / 0.25 |
| Interarm fraction | 0.7 | 1 |

The thin disk is about 3.33 times thicker than the previous single layer
(`diskThickness = 1.5`); the thick disk is three times thicker again. The latter
uses a warm color, no spiral concentration, and no highlights. It provides a
faint population above and below the bright plane rather than making every
high-latitude catalog star part of a uniformly bright band.

There are still 300,000 particles in total. The expected bulge population is
unchanged at 66,000 particles (`240000 * 0.275`, previously `300000 * 0.22`).
Only the thin-disk definition generates a bulge. Its vertical spread increases
from 1 to 2 generator units so that the central bulge remains thicker than the
bright disk; its horizontal spread, color, and opacity are retained. These are
random particle populations, so actual counts and appearance vary between loads.

Both layers retain the previous overall radius, object scale, render scale,
radial falloff, orientation, and visual rotation period. The main layer also
retains the previous spiral parameters and highlight probability.

## Interpretation and tuning

These settings are a visual approximation, not a measured Milky Way density fit.
The generator uses `y = Gaussian(0.22 * diskThickness)`. Neither `diskThickness`
nor `galaxyRadius` is expressed directly in parsecs. Their visible sizes also
depend on `objectScale` and the parent scaled-space transform. `renderScale`
cancels out of particle positions and sizes and should not be used to tune
visible thickness.

The real Galaxy has populations with different vertical distributions. For
context, a Gaia analysis finds local exponential scale heights near 260 pc for
the thin disk and 700 pc for the thick disk. Those exponential scale heights
are neither hard edges nor the Gaussian standard deviations used by this
renderer. See [Everall et al. (2022)](https://academic.oup.com/mnras/article/511/3/3863/6521448).
The appearance should be checked in KSP before claiming an absolute physical
thickness; no in-game visual validation was performed for this change.

The saved star catalog includes estimated heights approaching 2 kpc from its
Galactic plane. Some markers can remain outside the visible light distribution.
The catalog also favors the outward direction and is not a representative
sample of the Galaxy's density. Do not flatten or move those stars to match the
particle effect.

For further tuning, adjust `diskThickness` for vertical spread and the fourth
color component for opacity. Keep the two layer planes aligned with one another
and with the existing star-position converter. The total particle count is
unchanged, although a second particle system and transparent overdraw can still
affect performance.

## Loading and toolbar behavior

The local GalacticDisk startup loader creates a separate particle system for
each definition, even when their `planet` values match. Its toolbar dictionary
keeps only the last definition for each body, so the thin layer is deliberately
listed last. Edit both layers in the configuration file and restart KSP to load
both together; toolbar edits alone do not update both layers.
