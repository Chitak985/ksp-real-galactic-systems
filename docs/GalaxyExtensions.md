# Galaxy extension folders

Each generated galaxy's entry in `data/galaxies.json` has a local packaging choice named `far`:

```json
"far": "Far1"
```

| Saved value | Output folder |
| --- | --- |
| `Far1` | `RealGalacticSystems/Extension1` |
| `Far2` | `RealGalacticSystems/Extension2` |
| `Far3` | `RealGalacticSystems/Extension3` |
| Missing, null, `None`, or anything else | `RealGalacticSystems/Systems` |

This choice controls only the output location. It is not emitted into the game configuration and is independent of the galaxy's physical distance. Edit it directly to move a galaxy between packages, then run `python -B rebuild_galaxies.py`. Source-replay checks validate astronomical measurements separately from this local packaging metadata. New imports without an assignment use the base Systems folder; an ordinary refresh preserves an existing assignment unless explicitly overridden.

The initial assignments preserve the previous groups: 1,225 galaxies in Extension1, 463 in Extension2 and 47 in Extension3. The 322 other generated galaxies stay in Systems, alongside all eight manual galaxy hosts in their existing directories. Body and disk definitions remain together in the same file. File names, cache paths, body names, orbits, disk appearances and physical values do not change.

Both the rebuild command and galaxy import commands use the same relocation writer. A new destination is written before any previous generated copy is removed. Only files with the matching generator ownership header, galaxy ID and body name can be removed from an old output folder. Unmanaged files and identity conflicts cause an error. `--galaxy ID` limits relocation to the selected galaxy; `--check` reports obsolete copies and missing/outdated destinations; `--dry-run` lists the intended writes and removals without changing files. Removing a galaxy from the catalog does not delete its output automatically.

The folders can be packaged or omitted as separate extensions. This change does not modify the existing settings or loading-screen configuration.
