#!/usr/bin/env python3
"""Import measured regular galaxies in bulk, retaining sources and selection audits."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import fetch_galaxies as fetch
import fetch_nearby_galaxies as nearby
import rebuild_galaxies as builder
from rebuild_configs import CatalogError, load_catalog, write_configs

ROOT = Path(__file__).resolve().parent
SNAPSHOT = "data/galaxy_sources/simbad_bulk_continuation_snapshot.json"
REPORT = "data/bulk_galaxy_selection.json"
LEGACY_QUERY = ("SELECT TOP 4000 b.oid," + fetch.FIELDS +
         ",b.rvz_nature,b.rvz_qual,b.rvz_err,b.galdim_qual "
         "FROM basic AS b JOIN allfluxes AS f ON b.oid=f.oidref "
         "WHERE b.otype='G' AND b.rvz_redshift>=0.01 AND b.rvz_redshift<=0.1 "
         "AND b.galdim_majaxis>0 AND b.galdim_minaxis>0 "
         "AND b.galdim_angle IS NOT NULL AND b.morph_type IS NOT NULL "
         "AND f.B IS NOT NULL AND f.K IS NOT NULL ORDER BY rvz_redshift,oid")
QUERY = LEGACY_QUERY.replace("b.rvz_redshift>=0.01", "b.rvz_redshift>=0.02582", 1)
SUPPORTED_QUERIES = {LEGACY_QUERY, QUERY}
BULK_ENDPOINTS = ("https://simbad.idia.ac.za/simbad/sim-tap/sync", *fetch.SIMBAD)
NOTE = ("Distance assumes cosmological redshift; peculiar velocities add uncertainty. "
        "Stellar mass uses estimated Ks M/L=1 and excludes the dark halo. "
        "Isophotal size, 3D depth, colors, spiral phase and core gravity are approximate.")


def snapshot_record_path(snapshot):
    """Keep every imported response addressable after later live refreshes."""
    digest = snapshot.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CatalogError("Invalid source snapshot digest.")
    return f"data/galaxy_sources/simbad_bulk_{digest}.json"


def validate_snapshot(snapshot):
    if snapshot.get("schema_version") != 1 or snapshot.get("query") not in SUPPORTED_QUERIES:
        raise CatalogError("Unsupported bulk snapshot version or query.")
    snapshot_record_path(snapshot)
    text = snapshot["text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != snapshot["sha256"]:
        raise CatalogError("Bulk snapshot checksum mismatch.")
    document = json.loads(text)
    keys = [column["name"] for column in document["metadata"]]
    rows = [dict(zip(keys, row)) for row in document["data"]]
    if rows != snapshot["rows"] or len({r["oid"] for r in rows}) != len(rows):
        raise CatalogError("Snapshot rows differ from their original response or contain duplicate objects.")
    return rows


def normalize_row(row, snapshot, settings):
    if row.get("otype") != "G" or row.get("rvz_nature") not in {"s", "se", "sa"}:
        raise CatalogError("Requires a galaxy with an explicitly spectroscopic redshift.")
    if row.get("rvz_qual") not in {"A", "B", "C"} or row.get("galdim_qual") not in {"A", "B", "C"}:
        raise CatalogError("Redshift and dimension quality must both be A, B or C.")
    if not 0.01 <= row["rvz_redshift"] <= 0.1:
        raise CatalogError("Outside the bulk query's supported redshift interval.")
    if not all(row.get(key) for key in ("morph_bibcode", "galdim_bibcode", "rvz_bibcode")):
        raise CatalogError("Missing original measurement references.")
    if any(tag in row["morph_type"] for tag in ("?", ":", "/")):
        raise CatalogError("Ambiguous morphological classification.")
    builder.classify(row["morph_type"])
    if row.get("B") is None or row.get("K") is None or not -5 < row["B"] < 30 or not -5 < row["K"] < 30:
        raise CatalogError("Missing or invalid B/K photometry.")
    name = " ".join(row["main_id"].split())
    requested = name if re.match(r"[A-Za-z]", name) else "Galaxy " + name
    body_name = re.sub(r"[^A-Za-z0-9_]", "", name)
    if not re.match(r"[A-Za-z]", body_name):
        body_name = "Galaxy" + body_name
    raw = {"requested_name": requested, "retrieved_utc": snapshot["retrieved_utc"],
           "bulk_snapshot": snapshot_record_path(snapshot), "bulk_snapshot_sha256": snapshot["sha256"],
           "simbad": {"url": snapshot["url"], "rows": [row], "text": None,
                      "note": "Original response text and metadata are retained in bulk_snapshot."},
           "nearby": {"rows": [], "url": None, "text": None, "skipped": True}}
    overrides = {"body_name": body_name, "uncertainty_note": NOTE}
    body = fetch.normalize(raw, settings, overrides)
    radius_pc = builder.distance_measures(body, settings)["angular_diameter_mpc"] * 1e6 * math.tan(math.radians(body["angular_major_arcmin"]/120))
    if not 100 <= radius_pc <= 200000 or not 1e5 <= body["mass_solar"] <= 1e14 or body["axis_ratio"] < .05:
        raise CatalogError("Implausible physical size, stellar mass, or flattening; individual review required.")
    return body, raw


def existing_aliases(catalog):
    aliases = set(builder.EXCLUDED)
    for body in catalog["galaxies"]:
        aliases.update(nearby.canonical(body[key]) for key in ("id", "name", "body_name"))
    return aliases


def same_sightline(a, b):
    # An identity safeguard, not a cone match against an unrelated catalog.
    ra1, dec1, ra2, dec2 = map(math.radians, (a["ra_deg"], a["dec_deg"], b["ra_deg"], b["dec_deg"]))
    hav = math.sin((dec1-dec2)/2)**2 + math.cos(dec1)*math.cos(dec2)*math.sin((ra1-ra2)/2)**2
    return hav <= math.sin(math.radians(5/3600)/2)**2


def select_bulk(snapshot, catalog, count):
    rows = validate_snapshot(snapshot)
    aliases = existing_aliases(catalog)
    selected, skipped = [], []
    for row in sorted(rows, key=lambda r: (r["rvz_redshift"], r["oid"])):
        try:
            if nearby.canonical(row["main_id"]) in aliases:
                raise CatalogError("Already present or excluded.")
            body, raw = normalize_row(row, snapshot, catalog["settings"])
            if any(same_sightline(body, old) for old in catalog["galaxies"]) or any(same_sightline(body, old) for old, _ in selected):
                raise CatalogError("Within five arcseconds of an existing galaxy; identity needs review.")
        except (CatalogError, ValueError, KeyError, TypeError) as exc:
            skipped.append({"oid": row["oid"], "name": row["main_id"], "reason": str(exc)})
            continue
        selected.append((body, raw))
        aliases.update(nearby.canonical(body[key]) for key in ("id", "name", "body_name"))
        if len(selected) == count:
            return selected, skipped
    raise CatalogError(f"Only {len(selected)} new bulk galaxies pass the checks; requested {count}.")


def select_nearby(catalog, count):
    snapshot = load_catalog(ROOT/"data/galaxy_sources/lvdb_nearby_snapshot.json")
    review = load_catalog(ROOT/"data/nearby_galaxy_review.json")
    rows = nearby.snapshot_rows(snapshot, review)
    try:
        return nearby.select_nearest(rows, review, snapshot, catalog, count)
    except CatalogError as exc:
        match = re.fullmatch(r"Only (\d+) eligible new galaxies have sufficiently complete data; requested \d+\.", str(exc))
        if not match:
            raise
        available = int(match.group(1))
        return nearby.select_nearest(rows, review, snapshot, catalog, available) if available else ([], [])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.count <= 4000:
            raise CatalogError("count must be between 1 and 4000.")
        if args.offline:
            snapshot = load_catalog(ROOT/SNAPSHOT)
        else:
            result = fetch.request(BULK_ENDPOINTS, {"request": "doQuery", "lang": "adql", "format": "json", "query": QUERY}, timeout=55)
            document = json.loads(result["text"])
            keys = [column["name"] for column in document["metadata"]]
            snapshot = {"schema_version": 1, "database": "SIMBAD bulk", "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                        "query": QUERY, "url": result["url"], "text": result["text"],
                        "sha256": hashlib.sha256(result["text"].encode("utf-8")).hexdigest(),
                        "rows": [dict(zip(keys, row)) for row in document["data"]]}
        validate_snapshot(snapshot)
        catalog = load_catalog(ROOT/"data/galaxies.json")
        chosen, nearby_skipped = select_nearby(catalog, args.count)
        updated = copy.deepcopy(catalog)
        updated["galaxies"].extend(body for body, raw in chosen)
        remainder = args.count-len(chosen)
        bulk, bulk_skipped = select_bulk(snapshot, updated, remainder) if remainder else ([], [])
        chosen += bulk
        updated["galaxies"].extend(body for body, raw in bulk)
        updated["galaxies"].sort(key=lambda body: builder.distance_measures(body, updated["settings"])["placement_mpc"])
        for body, raw in chosen:
            updated["rendering_overrides"][body["id"]] = {
                "parameters": {"starCount": 2000},
                "notes": "Two thousand representative mesh stars limit catalog resource costs; diffuse structure is unchanged. Count is a display budget, not a measured stellar population.",
                "sources": ["docs/SecondThousandGalaxies.md"]}
        ids = [body["id"] for body, raw in chosen]
        configs = builder.build_configs(updated, ROOT, ids)
        report = {"schema_version": 1, "retrieved_utc": snapshot["retrieved_utc"], "snapshot": snapshot_record_path(snapshot),
                  "snapshot_sha256": snapshot["sha256"], "selected_ids": ids,
                  "requested_count": args.count,
                  "nearby_count": len(chosen)-len(bulk), "bulk_count": len(bulk),
                  "selection": "Remaining eligible LVDB galaxies first, then SIMBAD rows by ascending spectroscopic redshift and object ID. This is not a complete nearest-galaxy census.",
                  "nearby_skipped": nearby_skipped, "bulk_skipped": bulk_skipped}
        if (ROOT/REPORT).exists():
            prior = load_catalog(ROOT/REPORT)
            report["previous_selections"] = [*prior.pop("previous_selections", []), prior]
        print(f"Validated {len(chosen)} new galaxies: {report['nearby_count']} LVDB, {len(bulk)} SIMBAD; catalog total {len(updated['galaxies'])}.")
        if args.dry_run:
            print("No files saved.")
            return 0
        pending = {ROOT/body["raw_record"]: json.dumps(raw, indent=2, allow_nan=False)+"\n" for body, raw in chosen}
        pending.update({ROOT/"data/galaxies.json": json.dumps(updated, indent=2, allow_nan=False)+"\n",
                        ROOT/SNAPSHOT: json.dumps(snapshot, indent=2, allow_nan=False)+"\n",
                        ROOT/snapshot_record_path(snapshot): json.dumps(snapshot, indent=2, allow_nan=False)+"\n",
                        ROOT/REPORT: json.dumps(report, indent=2, allow_nan=False)+"\n"})
        write_configs(pending)
        write_configs(configs)
        print(f"Saved sources, selection audit and {len(configs)} new configs.")
        return 0
    except (CatalogError, OSError, ValueError, TypeError, KeyError) as exc:
        print("Cannot import bulk galaxies: "+str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
