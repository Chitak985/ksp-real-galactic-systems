# Remaining saved-source galaxies between 100 and 150 Mpc

Added all **655** eligible galaxies remaining in the saved SIMBAD continuation snapshot. Every addition is above 125 Mpc and at or below 150 Mpc in nominal comoving placement distance, so every record has `"far": "Far2"` and its configuration is written directly into `RealGalacticSystems/Extension2`. No new Far1 or Far3 assignments are needed for this batch.

The snapshot was retrieved on 2026-11-19 and is retained, with its original response and checksum, at `data/galaxy_sources/simbad_bulk_0837ea28eb1729c1a6676f4bc92e4766ff206fa71305c03508f20aadb0b05af2.json`. No fresh network query or new distance assumption was needed. The exact selection, source checksum, selected IDs, rejection audit and distance extrema are saved in `data/remaining_100_150_mpc_selection.json`. Each addition also has its own raw record linked from `data/galaxies.json`.

Selection uses the existing bulk importer gates: explicitly spectroscopic redshifts; redshift and size quality A, B or C; original measurement references; measured major/minor axes and position angle; usable B/K photometry; supported regular morphology; and plausible inferred size, mass and flattening. Existing names/aliases and sources within five arcseconds of an existing or newly selected galaxy are excluded. The snapshot is a selected catalog sample, not a complete census of galaxies in this distance interval.

Descriptions retain the uncertainty notes: distance assumes cosmological redshift and is affected by peculiar velocities; stellar mass estimates use Ks mass-to-light ratio 1 and exclude the dark halo; depth, colors, spiral phase and central gravity are approximate. Assignments follow the nominal saved distances; measurement uncertainty is not a guarantee that an object physically lies on one particular side of a package boundary.

Each galaxy uses 2,000 representative mesh stars, matching the existing bulk batches. These are display samples, not literal stellar populations. The existing templates provide the diffuse appearance, fixed sky orientation, compact core SOI and frozen Sun-parented positioning orbit. No existing galaxy configuration or manual body is changed.

After this batch, the generated catalog contains **2,712 galaxies**: 322 in Systems, 1,225 in Extension1, 1,118 in Extension2 and 47 in Extension3. All eight manual galaxies remain in their existing Systems paths, giving **2,720 galaxy bodies overall**.

These totals describe saved catalog assignments. Before this addition, the repository's Extension1 folder, the previous Extension2 files (1,688 earlier galaxy configs combined), and `RealGalacticSystems/Configs/Unknown.cfg` were absent. Those omissions are preserved. This batch adds only its 655 new Extension2 configs. Full-catalog and manual-position checks therefore still report missing older files; the new entries are checked against their saved source records and generated template output separately.

To rebuild the saved data, use `python -B rebuild_galaxies.py`; to verify it, use `python -B rebuild_galaxies.py --check`.
