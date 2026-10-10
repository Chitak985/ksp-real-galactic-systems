# Rebuilding stars

`data/bodies.json` is the saved catalog for generated bodies. It currently contains
Tau Ceti. This first version supports **single Sun-like stars in MilkyWay only**;
galaxies, binaries, and other existing configs are outside this catalog.

From the repository directory, run:

```powershell
python rebuild_configs.py
```

Python 3.10 or newer is sufficient. This script uses only the standard library:
there are no package installations or network requests when rebuilding.
It also works when launched from another directory using its full path.

Every catalogued star is written directly to:

```text
RealGalacticSystems/Systems/[Galaxy]/Systems/[lowercase_body_name].cfg
```

For example, Tau Ceti generates
`RealGalacticSystems/Systems/MilkyWay/Systems/tau_ceti.cfg`, with the internal
name `Tau_Ceti`. Every non-ASCII-letter/digit character in the common name becomes
an underscore. Conflicting names and reserved Windows filenames are rejected.

## Changing the shared template

Edit `templates/sun_like_star.cfg.template` and run the rebuild command again.
This is the supplied template with the commented custom values replaced by named
placeholders. Its filename ends in `.template`, so KSP does not load it as a config.

Unmarked settings and curves remain normal configuration text. Changing a setting
such as `contractWeight` or a curve key in this file changes every generated star
on the next rebuild. Generated `.cfg` files are overwritten, so make permanent
changes in the template or catalog.

`sunlightShadowStrength` is intentionally absent: Sun-like stars inherit KSP's
default, as requested.

The available placeholders are:

| Placeholder | Input or calculation |
| --- | --- |
| `{{body_name}}` | Common name with special characters replaced by underscores |
| `{{display_name}}` | Common name |
| `{{description}}` | Catalog description, fewer than 100 characters |
| `{{radius_m}}` | Catalog radius in meters |
| `{{mass_kg}}` | Catalog mass in kilograms |
| `{{rotation_period_seconds}}` | `rotation_period_days * 86400` |
| `{{luminosity}}` | `luminosity_solar * settings.sun_luminosity` |
| `{{temperature_k}}` | Catalog effective temperature, used for `ambientTemp` |
| `{{recovery_value}}` | `ceil(50 + distance_from_Sol_in_light_years)` (rounded up to a whole number) |
| `{{flying_altitude_threshold}}` | `radius_m / 10000` |
| `{{space_altitude_threshold}}` | `radius_m * 3` |
| `{{main_color}}` | Catalog RGBA rendering color |
| `{{ambient_color}}` | Main RGB divided by 10, with alpha retained |
| `{{orbit}}` | Complete generated frozen Orbit node |

`{{orbit}}` occupies its own line. All unmarked template values, including the
light and hazardous-body curves, are retained rather than recalculated.
`settings.sun_luminosity` defaults to the requested 1360 calibration; it is a
Kopernicus luminosity setting rather than a physical luminosity in watts.

## Adding or updating a star

Copy the Tau Ceti object in the `bodies` array, assign a unique `id`, and change
its name, description, physical values, and position. Keep `galaxy = MilkyWay`,
`template = sun_like`, and `system_type = single` for this version.

Positions use ICRS right ascension and declination **in degrees**, a distance
**in parsecs**, and the J2000 coordinate epoch. The catalog stores already selected
measurements; rebuilding does not look up data or propagate proper motion.
The `sources` array records references, uncertainties, conversions, and whether a
value is an author-chosen visual setting. It is documentation, not template text.

The template registry and catalog structure leave room for additional star
templates later, but unsupported types and binaries currently fail explicitly.
Removing a catalog entry does not delete an old generated config automatically.
Remove that config yourself if you intend to remove the body from the mod.

## Positioning around Sun

Stars use `referenceBody = Sun`, `eccentricity = 0`, `period = 3E300`,
and `mode = OFF`. Their hidden circular orbits encode positions, not observed
galactic motions. `meanAnomalyAtEpoch = 0` is correct here because the orbit's
orientation is calculated to place that anomaly at the requested position.

The converter rotates heliocentric ICRS coordinates into Galactic axes (x toward
the Galactic center, y toward longitude 90 degrees, z toward Galactic north).
These are treated as the right-handed orbital coordinate axes, with the orbital
xy plane corresponding to the rendered galaxy's plane. KSP's Unity world axes
have a different vertical-axis convention; the config contains orbital elements,
not a Unity Cartesian vector.

The resulting heliocentric position is used directly for the orbit around Sun.
The circular orbit's semi-major axis equals the star's distance from Sol; Tau Ceti
is about 11.91 light-years away. SagittariusA's configuration is not needed for
this calculation. Galactic axes determine the orientation of the placement,
while Sun remains the orbital parent.

This convention matches the current SagittariusA position along positive x and
the current unrotated GalacticDisk. A future change in the disk's orientation
would require changing this coordinate adapter as well.

Sun, SagittariusA, galaxy SOIs, and other existing bodies are not edited by this
script. The frozen stars no longer inherit SagittariusA's motion. This generator
does not implement the
earlier proposed SOI restructuring or establish an in-game thermal fix.

## Tau Ceti data

The physical data use [Korolik et al. (2023), Tables 3 and 4](https://arxiv.org/html/2307.10394v1):
radius 0.793 solar radii, mass 0.69 solar masses, temperature 5320 K,
luminosity 0.45 solar luminosities, and rotation period 46 days. Solar conversion
constants and the published uncertainties are saved in the catalog.

The position uses [SIMBAD](https://simbad.cds.unistra.fr/simbad/sim-id?Ident=tau+Cet)
ICRS J2000 coordinates and its Gaia EDR3 parallax of 273.8097 mas. The inverse
parallax gives approximately 3.65217 pc (11.9118 light-years). The supplied
rendering color is retained as an author-chosen color, not a measured RGB value.

## Checking the result

```powershell
python rebuild_configs.py --dry-run
python rebuild_configs.py --body tau_ceti
python rebuild_configs.py --check
python -m unittest discover -s tests -v
```

`--dry-run` validates and lists output paths without writing. `--check` compares
saved configs against the current catalog and templates and exits with code 1
if a file needs rebuilding. Invalid input exits with code 2. All selected bodies
are rendered and validated before writes start; each file is replaced atomically.
The script never deletes unmanaged configs or writes to Git.

The tests check physical/science conversions, catalog coordinates, orbital
round trips, propagation of template edits to multiple stars, and repeatable
output without altering unrelated files. KSP runtime behavior still requires
testing in the game.
