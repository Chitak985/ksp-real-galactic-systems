# Manual object positions and frozen Sun-parented orbits

All twelve hand-written objects are covered by `data/manual_object_positions.json`
and `rebuild_manual_orbits.py`. They use the same `frozen_orbit`,
`heliocentric_position` and cosmology functions as the generated objects. Each
Orbit is Sun-parented, circular, at epoch zero, with `period = 3E300` and
`mode = OFF`. The arbitrary circular orbit encodes a Cartesian position;
inclination and node values are not physical galactic orbital elements.
No real dynamical orbit is inferred for these objects.

Stock Sun remains the root and is never edited, preserving system-replacer
compatibility. No generated star or galaxy config is changed. The existing
manual templates, masses, radii, curves, luminosities, disk rotations, opacity
presets and visual properties are preserved. Triangulum already matches the
method and needs no file change; eleven manual configs change.

## Placement summary

Distances are nominal heliocentric values. Cosmological placements use present
comoving distances in a flat model; their lookback times are integrated
separately. A megaparsec is about 3.262 million light-years.

| Body / config | Placement (Mpc) | Lookback (Gyr) | Core SOI (pc) | Method |
|---|---:|---:|---:|---|
| [SagittariusA](../RealGalacticSystems/Systems/MilkyWay/Configs/SagittariusA.cfg) | 0.008277 | 0.00003 | 100 | icrs distance |
| [AndromedaA](../RealGalacticSystems/Systems/Andromeda/Configs/AndromedaA.cfg) | 0.777787 | 0.00254 | 100 | icrs distance |
| [TriangulumBarycenter](../RealGalacticSystems/Systems/Triangulum/Configs/TriangulumBarycenter.cfg) | 0.836121 | 0.00273 | 9.72234 | icrs distance |
| [LargeMagellanicCloud](../RealGalacticSystems/Systems/MilkyWay/LargeMagellanicCloud/Configs/LargeMagellanicCloud.cfg) | 0.049590 | 0.00016 | 100 | icrs distance |
| [SmallMagellanicCloud](../RealGalacticSystems/Systems/MilkyWay/SmallMagellanicCloud/Configs/SmallMagellanicCloud.cfg) | 0.062440 | 0.00020 | 100 | icrs distance |
| [NGC5866](../RealGalacticSystems/Systems/NGC5866/Configs/NGC5866.cfg) | 15.300000 | 0.04990 | 100 | icrs distance |
| [TheGreatAttractor](../RealGalacticSystems/Systems/TheGreatAttractor/Configs/TheGreatAttractor.cfg) | 64.540059 | Not inferred | 100 | galactic distance |
| [AE_Andromedae](../RealGalacticSystems/Systems/Andromeda/Systems/AE_Andromedae/Configs/AE_Andromedae.cfg) | 0.777787 | 0.00254 | 1 | icrs distance |
| [MGC1](../RealGalacticSystems/Systems/Andromeda/Systems/MGC1/Configs/MGC1Barycenter.cfg) | 0.616595 | 0.00201 | 5 | icrs distance |
| [GN-z11](../RealGalacticSystems/Systems/GN-Z11/Configs/GN-Z11Blackhole.cfg) | 9,754.639020 | 13.35680 | 5 | cosmological redshift |
| [JADES-GS-Z14-0](../RealGalacticSystems/Systems/JADES-GS-Z14-0/Configs/JADES-GS-Z14-0Barycenter.cfg) | 10,338.305341 | 13.50111 | 5 | cosmological redshift |
| [unknown](../RealGalacticSystems/Configs/Unknown.cfg) | Fictional; preserved | Not inferred | 3.24078e-07 | preserved orbit |

## Research and approximations

- **Sagittarius A:** measured ICRS direction and GRAVITY distance 8,277 pc.
  Its invisible orbit places the Galactic center relative to the unmodified
  Sun. The Milky Way disk stays in its existing Galactic world plane.
  [GRAVITY analysis](https://arxiv.org/abs/2112.07478).
- **Andromeda and Triangulum:** their reviewed directions and valid existing
  representative distances are retained. They remain static children of Sun.
  Sources are retained in `data/legacy_disk_orientations.json`.
- **LMC and SMC:** catalog centers and eclipsing-binary mean distances of
  49.59 and 62.44 kpc. Their irregular shapes, substantial depth and choice of
  center limit a point-object interpretation. [LMC](https://arxiv.org/abs/1903.08096),
  [SMC](https://arxiv.org/abs/2010.08754).
- **NGC 5866:** catalog position with the literature distance 15.3 Mpc,
  approximately 0.7 Mpc uncertainty. The old artificial orbit placed it much
  too close. [Distance table](https://doi.org/10.1088/0004-6256/144/6/164).
- **Great Attractor:** a representative historical flow-model center at
  Galactic longitude 307 degrees, latitude +9 degrees and distance 4350/H0 Mpc,
  or about 64.54 Mpc. The quoted 350 km/s radial-scale uncertainty alone is
  about 5.2 Mpc at the adopted H0. This extended overdensity has no unique
  compact center, and is distinct from a single Norma-cluster galaxy.
  [Regional analysis](https://academic.oup.com/mnras/article/369/3/1131/1227816).
- **AE Andromedae:** its Gaia/SIMBAD sky position with M31's adopted mean
  distance. A direct stellar parallax at this distance is not used. The star's
  depth within M31 remains unknown.
- **MGC1:** published M31-cluster coordinates and distance modulus 23.95 +/- 0.06,
  giving about 617 kpc with about 17 kpc uncertainty. It is roughly 200 kpc from
  M31, so its game parent becomes Sun. The bare SIMBAD identifier resolves to an
  unrelated star near RA 150 degrees; that match was explicitly rejected.
  [Mackey et al.](https://academic.oup.com/mnras/article/401/1/533/1008146).
- **Unknown:** a fictional object with no real astrometry. Its old Cartesian
  position at game time zero is preserved, including its intentionally extreme
  distance. Only the orbit representation is normalized.

The raw SIMBAD queries, original response text, coordinates, references and
checksums are saved under `data/manual_position_sources`. Each registry record
contains its adopted inputs, sources and uncertainty note. Paper inputs and
assumed distances are explicitly distinguished from catalog astrometry.

## Very distant galaxies

For **GN-z11**, adopt the JWST/MIRI [O III] redshift 10.6022 +/- 0.0004.
The H-alpha value is 10.6029 +/- 0.0006, and earlier NIRSpec spectroscopy gave
10.6034 +/- 0.0013. These small line/calibration differences matter more than
spurious numerical precision. [MIRI study](https://arxiv.org/html/2412.12826v1),
[NIRSpec study](https://doi.org/10.1051/0004-6361/202346159).

For **JADES-GS-z14-0**, adopt the ALMA [O III] redshift 14.1793 +/- 0.0007 and
Table 1 coordinates from Schouws et al. An independent reduction finds
14.1796 +/- 0.0007. Both supersede the broader original NIRSpec-break estimate.
[Schouws et al.](https://arxiv.org/html/2409.20549v2),
[Carniani et al.](https://arxiv.org/abs/2409.20533).

The existing cosmology uses H0=67.4 km/s/Mpc, Omega_m=0.315,
Omega_r=0.000092, and a flat cosmological constant component. It gives:

| Object | Comoving placement | Lookback time | Angular-diameter distance |
|---|---:|---:|---:|
| GN-z11 | 9,754.639 Mpc (~31.815 billion ly) | 13.3568 Gyr | 840.758 Mpc |
| JADES-GS-z14-0 | 10,338.305 Mpc (~33.719 billion ly) | 13.5011 Gyr | 681.079 Mpc |

Comoving distance supplies the static position in the same convention as the
rest of the mod. It is not luminosity distance or lookback time multiplied by c.
The game does not simulate cosmic expansion or evolve these observed galaxies
forward to their unknown present-day appearance. Small redshift uncertainties
do not capture uncertainty in the cosmological model, distance scale, line
centroids or foreground-lensing astrometry. No new physical size, luminosity or
black-hole mass is inferred or changed here.

The integration was checked independently using 128-point Gauss-Legendre
quadrature: relative differences from the shared Simpson implementation are
below 9e-11 for both comoving distance and lookback at these redshifts. Numerical
integration error is negligible compared with the scientific assumptions.

## Core boundaries and hierarchy

Manual SOIs are explicit small **game boundaries**, not measured gravitational
influence radii. Sagittarius A, Andromeda, the Magellanic Clouds, NGC 5866 and the
Great Attractor use 100 pc; Triangulum retains its existing 3E17 m boundary.
GN-z11, JADES and MGC1 use 5 pc, AE And uses 1 pc, and Unknown retains 1E10 m.
This removes automatic huge parent-relative SOIs for AE And and GN-z11 and
prevents several old manual SOIs from enclosing Sol. Stars or clusters with
hundred-million-year galactic motions are placed directly under Sun; no nested
M31 or Milky Way galactic orbit is retained here.

All manual SOIs exclude Sol and do not overlap each other or the 2,057 generated
galaxy core boundaries. These changes do not prove that every SOI or heating
problem elsewhere in the mod is solved. The scope is manual placement and core
isolation; the existing physical and lighting models remain as supplied.

## Rebuild and verify

```powershell
python rebuild_manual_orbits.py --check
python rebuild_manual_orbits.py --dry-run
python rebuild_manual_orbits.py
```

The updater edits each registered Orbit node and the explicit
`Properties/sphereOfInfluence` value. Hand-written galaxies
keep their existing paths in the base Systems folder. It preserves hand-written line endings and
cosmetic orbit colors/camera settings. It refuses a Sun target, a generated
config, an identity mismatch, invalid coordinates or an SOI enclosing Sol.
Repeat execution is idempotent. Run the normal star and galaxy rebuild scripts
separately for their catalogs.


## Verification and limits

All 76 automated tests pass. Seven manual-placement tests cover complete
hand-written-body coverage, Sun protection, frozen position recovery, independent
ICRS/Galactic coordinate recovery, high-redshift distance distinctions, fictional
position preservation, compact SOI isolation, identity checks and idempotence.
The declared-coordinate-frame rejection was also checked after the final guard
was added. Existing generated-body, source, orientation and template tests pass.

Every modified manual config is compared with its original after masking only
the Orbit node and explicit SOI field. All other content is unchanged. Generated
galaxy/star configs and README are compared byte for byte. All three rebuild
check modes validate their own saved data. No live KSP session, user save, or
rendering/physics integration at the new positions was tested. Existing cache
files were not removed; the configuration remains the authoritative input.
