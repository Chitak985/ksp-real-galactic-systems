# Fetching and rebuilding galaxies

`fetch_galaxies.py` downloads public catalog records and adds galaxies to the mod.
`rebuild_galaxies.py` rebuilds them from saved JSON and editable templates without
network access. Both require Python 3.10+ and only the standard library. The
existing Milky Way star generator remains separate.

The generated bodies require Kopernicus and a GalacticDisk version with its
**Volumetric** renderer and shader asset bundle. The new tool uses the settings
supported by the local GalacticDisk source and `ExampleVolumetricConfig.txt`.
It does not install or modify GalacticDisk itself.

## Commands

Run these from the repository directory. Quoting each galaxy name preserves its
spaces. Fetching normally saves the adopted catalog, the original responses,
and the generated configuration files:

```powershell
python fetch_galaxies.py "NGC 147" "NGC 185" "NGC 300" "NGC 404" "NGC 2403"
python rebuild_galaxies.py
python rebuild_galaxies.py --check
python rebuild_galaxies.py --dry-run
python rebuild_galaxies.py --galaxy ngc_300
```

Existing adopted entries are retained unless `--refresh` is supplied. A refresh
replaces their adopted values, so save any reviewed corrections in an overrides
file and pass that file again when refreshing. These additional fetch modes are
available:

```powershell
python fetch_galaxies.py "NGC 300" --refresh --dry-run
python fetch_galaxies.py "NGC 300" --offline --refresh --dry-run
python fetch_galaxies.py "NGC 300" --catalog-only
python fetch_galaxies.py "NGC 1365" --simbad-only --overrides data/spiral_galaxy_adoptions.json
```

`--offline` uses saved raw responses and makes no requests. `--dry-run` writes
nothing, including caches. `--catalog-only` saves the data but leaves game configs
for the separate rebuild command. `--simbad-only` explicitly omits the nearby query, records this in provenance,
and requires alternate reviewed inputs where needed. It does not convert query
errors into empty matches.

`--catalog PATH` chooses another adopted JSON
file; templates and game output paths still resolve against the script directory.
Every selected galaxy is validated before the importer starts writing. Each file
is replaced atomically; the multi-file operation is not a database transaction.

All output configurations are **files directly under**
`RealGalacticSystems/Systems`, such as `ngc_300.cfg`. No per-galaxy directories are
created. Files without this generator's ownership header, existing body names,
duplicate entries, and unsafe paths are protected. Removing a catalog entry does
not delete its old configuration automatically.

## Data and provenance

The saved inputs are:

- `data/galaxies.json`: adopted positions, distances, morphologies, angular shapes,
  redshifts, masses, provenance, and uncertainty notes.
- `data/galaxy_sources/*.json`: original SIMBAD/VizieR responses, successful
  request URLs, and UTC retrieval timestamps. These allow offline reproduction.

SIMBAD supplies J2000 ICRS coordinates, morphological classifications, galaxy
position angles, redshifts, and fallback angular dimensions/photometry. The
VizieR **Updated Nearby Galaxy Catalog**, `J/AJ/145/101/catalog`, supplies
redshift-independent nearby distances, Holmberg angular diameters, axial ratios,
Hubble stages, and indicative masses or Ks luminosities. A 60-arcsecond cone
cross-match must identify one catalog object; ambiguous matches are rejected.

The importer tries the main CDS servers and then their Harvard mirrors. The main
servers were unavailable during the initial import; all five additions were
successfully fetched from the Harvard mirrors on 2026-11-15. This is a snapshot
of published catalog adoptions, including the 2013 nearby catalog, not a claim
that each adopted measurement is the newest available measurement.

References: [SIMBAD query documentation](https://simbad.u-strasbg.fr/Pages/guide/sim-q.htx),
[Karachentsev, Makarov & Kaisina (2013)](https://arxiv.org/abs/1303.5328), and the
[VizieR catalog](https://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/145/101).

## Templates

`templates/galaxies/base.cfg.template` preserves the supplied body and disk
skeleton. `{{disk_parameters}}` replaces its original `...` location. The
common renderer settings live in `common.disk.template`; each family file
includes them through `{{common_parameters}}` and adds its morphology settings.
All template filenames end in `.template`, so KSP does not load them.

| Template | Regular galaxy forms represented |
| --- | --- |
| `elliptical` | E0-E7, giant elliptical and cD envelopes as a smooth approximation |
| `compact_elliptical` | cE and ultracompact dwarf spheroids |
| `dwarf_elliptical` | dE and nucleated/emission dwarf elliptical classifications |
| `dwarf_spheroidal` | dSph/Sph diffuse spheroids |
| `lenticular` | SA0/S0, dwarf S0, and S0/a transitions |
| `barred_lenticular` | SB0/SAB0 |
| `spiral` | Ordinary/unbarred Sa-Sd and intermediate Hubble stages; dwarf/LSB size and brightness variants |
| `weakly_barred_spiral` | SABa-SABd |
| `barred_spiral` | SBa-SBd |

Spiral pitch, arm fraction, and bulge fraction vary with the adopted Hubble stage.
These are visual estimates, not resolved fits to each galaxy's arms. Bar angle,
arm phase, winding sense, detailed dust and nebula geometry are not inferred from
the available measurements. The generic templates approximate their family.
Spheroids use the renderer's ellipsoidal Gaussian, not a fitted Sersic profile;
their line-of-sight depth is unknown. Their `bulgeFraction = 1` means the whole
spheroid, not a second central galaxy or an extra body.

Irregular/Magellanic forms, ring-bearing classifications, merging/interacting
classifications, and explicitly peculiar morphologies are rejected by automatic
classification. Resonance-ring classifications are conservatively excluded too.
The tool does not invent regular structures for an unrecognized type. A reviewed
explicit `template` override can resolve conflicting classifications; it should
not be used to disguise an unsupported morphology.

Editing the base template affects every galaxy on the next rebuild. Editing a
family template affects its members. Palette values and renderer quality can
also be edited directly in these templates. The numeric geometry placeholders
include `object_scale`, `rotation_x/y/z`, `spheroid_sigma`, and
`spheroid_minor_sigma`; `mass_kg`, `gravity_g`, `soi`, `orbit`, `galaxy_name`,
`display_name`, and `description` fill the base body. Unknown placeholders and
duplicate generated keys fail validation.

## Position, apparent shape, and light travel

Positions use the **same Galactic-axis adapter as the existing stars**. Every
galaxy orbits the stock `Sun` with a circular placement orbit, `period = 3E300`,
and `mode = OFF`. No other central body is created. The rendered galaxy is
attached directly to its one invisible Jool-derived barycenter. Disk rotation
is static: `alignRotationToOrbit = false` and `rotationPeriod = 0`.

The observed position angle is measured east of celestial north. Disk
inclination uses reviewed measurements where available, otherwise it is
inferred from the observed axial ratio assuming intrinsic
flattening 0.15 for spirals and 0.25 for lenticulars. This has the usual near-side
ambiguity. Spheroids reproduce the measured sky ellipticity while assuming an
unmeasured depth. The tool converts the complete sky basis to Unity Euler
rotations, accounting for KSP's orbital-to-Unity Y/Z swap.

Reviewed `inclination_deg` and `position_angle_deg` values can be saved alongside
`parameters`, `notes`, and `sources` in `rendering_overrides`. They survive data
refreshes and template rebuilds. See [disk orientations](DiskOrientations.md)
for all current adoptions, legacy galaxy corrections, and remaining ambiguities.

The normalized `galaxyRadius = 100` and computed `objectScale` reproduce the
observed angular major diameter from Sol. Volumetric rendering cancels the
parent body's scale; the default KSP scaled-space conversion is 6000. Catalog
diameters are isophotal outlines, so a smooth light profile is not a sharp edge.

**Observations already show the galaxy at its light-emission epoch.** Their
morphology and position are used as observed, without advancing to its imagined
present state or rewinding the same structure a second time. Descriptions include
an approximate lookback time. Local redshift-independent distances use D/c;
peculiar velocities and negative redshifts never become negative distances or
negative ages.

`distance.kind = adopted_local` also accepts published local compilations that
may mix direct measurements, group membership, and flow models. The method and
uncertainties are saved rather than labeling every estimate redshift-independent.
It uses the same approximate D/c lookback; cosmological redshift remains a
separate distance kind.

For a source without a nearby-catalog distance and with z >= 0.01, an explicitly
assumed cosmological redshift can supply flat-FLRW distances and lookback time.
The defaults are H0=67.4 km/s/Mpc, Omega_m=0.315, Omega_r=0.000092, with the
remaining density in a cosmological constant. Position uses comoving distance;
rendered radius is adjusted to preserve the observed angular size. This is a
static Euclidean game layout, not an expanding-universe simulation. The mass and
physical size and SOI calculations use the angular-diameter distance;
luminosity-based mass estimates use luminosity distance.

Template population colors are approximate blackbody-equivalent palettes with
observer temperature T/(1+z); an optical H-alpha visibility approximation fades
the generic nebula color when shifted beyond the visible band. They are not a
reconstructed spectral energy distribution or an infrared false-color image.
Observed B magnitudes set a relative surface-brightness factor and **already
include propagation/redshift losses**, so no second cosmological dimming factor
is applied to those measurements. If B photometry is absent, the estimate uses
(1+z)^-4 instead. Brightness is intentionally bounded for the renderer and is not
an absolute photometric calibration. Profile-dependent display exposure is then
applied: a thin disk, lenticular bulge and full spheroid integrate different
amounts of light in the volume shader. The display response compresses the
catalog surface-brightness range and reduces additive mesh-star glare. Old
populations use warmer palettes than young spiral arms. These choices represent
a photographic presentation, not a prediction of naked-eye color or exposure.
Existing Milky Way foreground effects can still affect visibility from elsewhere
in the game.

Distance conventions and dimming follow
[Hogg, Distance Measures in Cosmology](https://ned.ipac.caltech.edu/level5/Hogg/Hogg_contents.html).

## Mass and core SOI

The mass is the nearby catalog's **indicative mass inside the Holmberg radius**
when present. Otherwise it is explicitly estimated from Ks luminosity with a
stellar mass-to-light ratio of 1 in solar units, excluding the unmeasured dark
halo. A SIMBAD Ks magnitude can provide a further fallback using an assumed
solar absolute Ks magnitude of 3.28. These are broad mass proxies, not measured
central black-hole masses.

The barycenter is a game point-mass representation. It does not reproduce the
galaxy's distributed gravitational field. Its SOI radius is 2% of the physical
optical radius, bounded to 5-100 pc; input is rejected if those bounds would
enclose 10% or more of a very small galaxy. The surrounding luminous galaxy is
outside this small SOI. `radius = 1` is an invisible proxy radius, not a physical
galaxy size. `geeASL = G * mass / (9.80665 * radius^2)` is kept consistent with
the supplied mass. This does not fix pre-existing KSP SOI or thermal problems.

## Reviewed corrections

For a missing or superseded measurement, create a JSON mapping the exact
requested name to reviewed fields, then use `--refresh --overrides FILE`:

```json
{
  "NGC 300": {
    "template": "spiral",
    "uncertainty_note": "Catalog distances and intrinsic disk thickness are uncertain; colors and core gravity are estimates."
  }
}
```

Allowed corrections include `distance`, `mass_solar`, `mass_kind`,
`angular_major_arcmin`, `axis_ratio`, `position_angle_deg`, `morphology`,
`hubble_t`, `redshift`, `b_magnitude`, `template`, `body_name`, `name`, and
`uncertainty_note`. If changing a distance, review distance-dependent catalog
masses as well and supply a corrected `mass_solar`; the importer does not refit
the catalog's original dynamics. Corrections are recorded in `adoption_notes`.
Uncertainty notes are included in the game's body description.

### Persistent appearance refinements

The optional top-level `rendering_overrides` object in `data/galaxies.json` holds
reviewed refinements by galaxy ID. It is separate from the fetched measurements
and survives `fetch_galaxies.py --refresh`. Each entry can contain:

- `inclination_deg`: a reviewed disk inclination, instead of the generic estimate
  from axial ratio. Position angle and the orbit remain catalog-derived.
- `parameters`: bounded numeric/boolean GalacticDisk appearance fields, such as
  `dustOpacity`, `armWidth`, `interarmFraction`, or `volumeIntensity`. They take
  precedence over family-template defaults. Protected placement and body fields
  cannot be overridden here. Remove an individual refinement to use its template
  default again.
- `notes` and `sources`: provenance and uncertainty for the refinements.

NGC 300 uses a measured 42-degree inclination and a weak bulge, with estimated
ragged arms and less opaque, smaller dust structures. The inclination is from
[Kruijssen et al. (2019), Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6544524/).
The weak bulge is consistent with
[Vlajic et al. (2009)](https://doi.org/10.1088/0004-637X/697/1/361).
NGC 404 uses the warmer, lower-exposure lenticular profile. Its distinct nuclear
cluster is not separately resolved by the generic template; see
[Seth et al. (2010)](https://arxiv.org/abs/1003.0680).

The NGC 300 reference photograph combines broad optical bands and emission-line
filters with roughly 50 hours of exposure. It is not a unique, unprocessed
color standard: [ESO image details](https://www.eso.org/public/images/eso1037a/).
Likewise, image crop, camera distance, and field of view change the apparent
screen size of NGC 404. The rendering correction does not shrink its measured
optical diameter to mimic a differently framed photograph.

## Initial five additions

The catalog now also includes [thirty spiral galaxies](SpiralGalaxies.md).

The catalog also includes [five very large galaxies](GiantGalaxies.md), added
using SIMBAD and reviewed stellar-size/mass measurements. It now also includes
[ten further large elliptical galaxies](LargeEllipticalGalaxies.md), with reviewed
optical diameters and photometric corrections. Their documentation
distinguishes deep stellar extents from ordinary catalog isophotes.

| Galaxy | Adopted family | Distance (Mpc) | Approximate lookback (Myr) | Core SOI radius (pc) |
| --- | --- | ---: | ---: | ---: |
| NGC 185 | Dwarf elliptical | 0.61 | 1.99 | 21.8 |
| NGC 147 | Dwarf spheroidal (catalog classification) | 0.76 | 2.48 | 29.8 |
| NGC 300 | Spiral, late stage | 2.15 | 7.01 | 100 |
| NGC 404 | Lenticular | 3.05 | 9.95 | 30.8 |
| NGC 2403 | Spiral, late stage | 3.18 | 10.37 | 100 |

NGC 147 uses the explicitly estimated Ks stellar-mass fallback; the other four
use catalog M26 indicative masses. Classification references can disagree:
for example, NGC 147 is commonly called a dwarf elliptical, while this catalog
labels its dwarf morphology Sph. The saved data retains both source records.

Milky Way, Andromeda, Triangulum, NGC5866, GN-z11, and JADES-GS-Z14-0 and their
common catalog aliases are excluded from this importer. The importer does not
modify their hand-authored files. A subsequent [orientation review](DiskOrientations.md)
corrected the Milky Way, Andromeda, and Triangulum disks separately. The star
catalog, Sun, and README remain untouched.

## Verification

```powershell
python -m unittest discover -s tests -v
python rebuild_galaxies.py --check
```

Tests cover sky-orientation projection, Galactic placement round trips, angular
sizes, local blueshifts, cosmological distances against an analytic limit,
template propagation, every regular family, source replay, mirror fallback,
offline dry runs, selected rebuilds, and protection of existing configurations.
KSP loading and appearance still require in-game verification.

The appearance correction was additionally checked using the actual local
GalacticDisk C# renderer and its packaged shaders in an isolated Unity 2019.4
preview, with a camera along the Sol-to-galaxy direction. Before/after previews
reproduced the near-invisible spiral and the pale lenticular core, and confirmed
visible spiral structure and a subdued warm bulge after correction. All five
generated galaxies were inspected. These checks use KSP API substitutes and no
Milky Way foreground, bloom, or user save; they are not a live KSP play test.


## Nearby spheroidal galaxies

[Fifty nearby galaxies](NearbyGalaxies.md) adds a pinned, checksum-verified Local
Volume Database importer: `fetch_nearby_galaxies.py`. It selects nearest new,
confirmed spheroidal systems with enough measured data, preserves raw records
and a selection audit, and supports `--offline` and `--dry-run`. Its documented
scope complements `fetch_galaxies.py`; neither importer accepts irregular, ring,
or merging morphologies. See the batch document for the list, source definitions,
uncertainties and commands.

The compiler accepts measured `v_magnitude` when `b_magnitude` is unavailable.
It keeps B as the first choice for existing records. Nearby V magnitudes are
explicitly marked extinction-corrected; foreground reddening is not recreated.
`spheroid_extent_sigma` optionally sets the Gaussian model boundary (2-6 sigma,
spheroid families only). Its default is 3.5, preserving existing output. The
nearby importer adopts 3 and converts measured half-light semi-major radii so
their projected sizes and axis ratios are preserved.


## Bulk measured-galaxy import

[The 900-galaxy batch](BulkGalaxies.md) brings the generated catalog to 1,000.
`fetch_bulk_galaxies.py` combines remaining eligible LVDB systems with explicit
spectroscopic SIMBAD records, retaining a response snapshot, per-object records
and a selection audit. `--offline` uses the saved response; `--dry-run` validates
without writes. Each new object has an explicit `starCount = 2000` rendering
override to limit mesh costs; original objects and template defaults are preserved.
See the batch document for sources, selection limits, commands, estimates and
remaining live-game performance checks.


## Second thousand and preserved source snapshots

[Another 1,000 galaxies](SecondThousandGalaxies.md) brings the generated catalog
to 2,000. The bulk importer now defaults to 1,000 additions, uses a bounded
continuation query starting at z=0.02582, and preserves every new original
response under a checksum-based filename. The old 900-galaxy snapshot and audit
remain available. Repeated distance integrals are cached in process without
changing output; see the batch document for the list, source limits and tests.


## Reviewed missing velocities and selected distant additions

[The 57-galaxy batch](FarAndNearbyGalaxies.md) brings the generated catalog to
2,057. `add_reviewed_galaxies.py` imports fifteen explicitly reviewed nearby
candidates and saved-source galaxies beyond 150 Mpc. Nearby unknown redshifts
remain null and use an unshifted display palette; cluster and flow-model proxies
are labeled. These modes require local distances. Raw LVDB rows remain intact
while separate literature-review blocks record measurement updates and sources.
Default strict imports and original generated configs retain their behavior.
