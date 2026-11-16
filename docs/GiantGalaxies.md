# Five very large galaxies

Added on 2026-11-16 using the galaxy importer. The saved galaxy catalog now has
**10 entries**. The new configs are files directly in
`RealGalacticSystems/Systems`, alongside the first five generated galaxies.

The selection prioritizes large **stellar bodies**, adequate data, and regular
elliptical/spiral morphology. It is not a claim that these are the five largest
galaxies in a uniform survey. A galaxy's quoted size depends strongly on bandpass,
surface-brightness depth, and whether diffuse intracluster light is included.
Gas disks and radio lobes were not used as stellar diameters. Strongly distorted
interacting systems such as NGC 6872 were not selected; Malin 1's interactions
and unusually complex faint structure were also left outside this batch.

## Adopted sizes

These are the approximate physical stellar diameters produced by the saved
angular sizes and the generator's configured cosmology. They are not comparable
at an identical limiting isophote. In particular, IC 1101 and UGC 1382 use deeper
literature extents, while the other three use SIMBAD catalog isophotes.

| Galaxy | Template | Angular major diameter | Model stellar diameter | Approx. light lookback |
| --- | --- | ---: | ---: | ---: |
| IC 1101 | Elliptical | 6.00 arcmin | 550 kpc / 1.79 million ly | 1,067 Myr |
| UGC 1382 | Spiral with a bright lenticular center | 7.00 arcmin | 172 kpc / 560,000 ly | 278 Myr |
| UGC 2885 (Rubin's Galaxy) | Spiral | 5.50 arcmin | 137 kpc / 448,000 ly | 283 Myr |
| NGC 6166 | Elliptical/cD main body | 2.34 arcmin | 91 kpc / 296,000 ly | 442 Myr |
| NGC 4889 | Elliptical | 2.92 arcmin | 79 kpc / 258,000 ly | 307 Myr |

SIMBAD redshifts drive the existing flat-FLRW distance and lookback calculation.
These redshifts include peculiar velocities, especially for cluster members.
Observed angular sizes are retained; the quoted physical sizes differ slightly
from papers that use a different cosmology. All new bodies keep frozen
Sun-parented placement, `period = 3E300`, `mode = OFF`, one invisible barycenter,
and a 100-pc core SOI. Their game descriptions mention the relevant uncertainties.

## Literature adoptions and limitations

**IC 1101:** the adopted 180-arcsecond semi-major-axis edge represents the main
stellar body. The 2026 analysis reports an edge radius near 260 +/- 38 kpc and
about 3.4 trillion solar masses in stars within that edge. The still larger
diffuse cluster-light region is omitted. The observed six-arcminute diameter
becomes about 550 kpc in the generator's cosmology. A single Gaussian ellipsoid
approximates the body; it does not reproduce the depleted nucleus or isophote
twists. [Deep imaging analysis](https://arxiv.org/html/2607.15340v1).

**UGC 1382:** the outer optical/UV stellar disk is detected to a radius of 3.5
arcminutes, around 80 kpc in the paper. This is distinct from the still larger
110-kpc-radius HI disk. Adopted stellar mass is 80 billion solar masses. Its
bright central lenticular component and very faint outer spiral are approximated
with a concentrated, dust-poor profile; the published multi-component Sersic fit
is not reproduced exactly. [Hagen et al. (2016)](https://arxiv.org/html/1607.02147v1).
The 46-degree inclination follows the compiled observational properties in
[Saburova et al. (2021)](https://academic.oup.com/mnras/article/503/1/830/6132255);
the adopted outer axial ratio is derived with intrinsic flattening 0.15.

**UGC 2885:** use the SIMBAD 5.5-arcminute stellar diameter and the 65-degree
inclination tabulated in the recent multiwavelength study. The adopted stellar
mass, 158 +/- 73 billion solar masses, is its Table 1 value; other published mass
estimates differ. Its weak resonance pseudoring is not modeled as a separate
structure; the supported model represents its ordinary spiral disk. This is not
a collisional ring-galaxy template.
[Multiwavelength study, Table 1](https://arxiv.org/html/2410.16467v1).

**NGC 6166 and NGC 4889:** use SIMBAD main-body optical isophotes, rather than
inflating their sizes to include entire clusters. Masses are explicitly the
importer's Ks luminosity estimates with stellar M/L=1, not black-hole masses or
cluster dynamical masses. NGC 6166's extended halo becomes controlled by cluster
gravity; the template omits that component and neighboring galaxies.
[Bender et al. (2015)](https://doi.org/10.1088/0004-637X/807/1/56),
[McConnell et al. (2011)](https://arxiv.org/abs/1112.1078).

## Reproducing and updating the additions

Raw responses are saved in `data/galaxy_sources`. The adopted physical values are
in `data/galaxies.json`. `data/giant_galaxy_adoptions.json` records the reviewed
replacement fields and source references; its `_references` object is metadata,
not an additional galaxy. Use that file again when explicitly refreshing:

```powershell
python fetch_galaxies.py "IC 1101" "UGC 1382" "UGC 2885" "NGC 4889" "NGC 6166" --refresh --overrides data/giant_galaxy_adoptions.json
python rebuild_galaxies.py --check
```

Add `--offline` to the fetch command to replay the saved responses. Rendering
refinements in the catalog are preserved during refresh. The importer now accepts
VizieR's metadata-only empty-table response, allowing the cosmological fallback
to run for galaxies outside the nearby catalog; malformed/error responses still
fail. No nearby-catalog measurements are attributed to those empty queries.

The previous five galaxy records, their rendering refinements, all star data and
configs, Sun, SagittariusA, and README are preserved. See
[GalaxyGeneration.md](GalaxyGeneration.md) for the shared workflow.

Verification: all 37 Python tests pass; all 10 galaxy configurations and all 150
star configurations match their saved inputs. The five new galaxies were also
visually inspected using the actual GalacticDisk renderer and packaged shaders
in an isolated Unity preview from the Sol viewing direction. This uses KSP API
substitutes and does not open a live KSP save. The very faint outer stellar
components are deliberately not rendered as uniformly bright disks.
