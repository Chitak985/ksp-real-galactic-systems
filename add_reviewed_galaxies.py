#!/usr/bin/env python3
"""Add reviewed nearby candidates and unused saved-source galaxies beyond 150 Mpc."""
import argparse
import copy
import json
from pathlib import Path
import sys

import fetch_bulk_galaxies as bulk
import fetch_nearby_galaxies as nearby
import rebuild_galaxies as builder
from rebuild_configs import CatalogError,load_catalog,write_configs

ROOT=Path(__file__).resolve().parent
REVIEW="data/reviewed_nearby_candidates.json"
REPORT="data/far_near_galaxy_selection.json"


def candidates(catalog,root=ROOT):
    review=load_catalog(root/REVIEW)
    source_review=load_catalog(root/"data/nearby_galaxy_review.json")
    source=load_catalog(root/"data/galaxy_sources/lvdb_nearby_snapshot.json")
    rows={row["key"]:row for row in nearby.snapshot_rows(source,source_review)}
    aliases=bulk.existing_aliases(catalog)
    chosen=[]
    for key,literature in review["candidates"].items():
        row=rows[key];meta=source_review["metadata"][key]
        if any(nearby.canonical(name) in aliases for name in [key,row["name"],*meta.get("aliases",[])]):
            continue
        if key in source_review["excluded"]:
            raise CatalogError("A reviewed exclusion needs independent reconsideration: "+key)
        raw={"schema_version":1,"database":"LVDB","version":source["version"],
             "csv_url":source["csv_url"],"csv_sha256":source["csv_sha256"],
             "retrieved_utc":source["retrieved_utc"],"row":row,"metadata":meta,
             "metadata_url":source_review["metadata_base_url"]+key+".yaml",
             "assumptions":{"extent_sigma":source_review["extent_sigma"]},
             "literature_review":literature}
        body=nearby.normalize_lvdb(raw,catalog["settings"])
        if any(bulk.same_sightline(body,old) for old in catalog["galaxies"]) or any(bulk.same_sightline(body,old) for old,_ in chosen):
            raise CatalogError("Reviewed galaxy identity needs inspection: "+key)
        chosen.append((body,raw))
        aliases.update(nearby.canonical(body[field]) for field in ("id","name","body_name"))
    updated=copy.deepcopy(catalog)
    updated["galaxies"].extend(body for body,raw in chosen)
    snapshot=load_catalog(root/bulk.SNAPSHOT)
    remaining,skipped=bulk.select_bulk(snapshot,updated,None)
    far=[(body,raw) for body,raw in remaining if builder.distance_measures(body,catalog["settings"])["placement_mpc"]>150]
    return chosen,far,snapshot,skipped


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    try:
        catalog=load_catalog(ROOT/"data/galaxies.json")
        near,far,snapshot,skipped=candidates(catalog,ROOT)
        chosen=near+far
        if not chosen:
            print("All reviewed candidates and saved-source galaxies beyond 150 Mpc are already present.")
            return 0
        updated=copy.deepcopy(catalog)
        updated["galaxies"].extend(body for body,raw in chosen)
        updated["galaxies"].sort(key=lambda body:builder.distance_measures(body,updated["settings"])["placement_mpc"])
        for body,raw in chosen:
            updated["rendering_overrides"][body["id"]]={"parameters":{"starCount":2000},
                "notes":"Representative mesh-star sampling budget; diffuse structure is retained.","sources":["docs/FarAndNearbyGalaxies.md"]}
        ids=[body["id"] for body,raw in chosen]
        configs=builder.build_configs(updated,ROOT,ids)
        report={"schema_version":1,"selected_ids":ids,"nearby_ids":[body["id"] for body,raw in near],
                "far_ids":[body["id"] for body,raw in far],"nearby_review":REVIEW,
                "far_min_comoving_mpc_exclusive":150,"far_snapshot":bulk.snapshot_record_path(snapshot),
                "far_snapshot_sha256":snapshot["sha256"],"far_source_rejections":skipped}
        if (ROOT/REPORT).exists():
            previous=load_catalog(ROOT/REPORT)
            report["previous_selections"]=[*previous.pop("previous_selections",[]),previous]
        print(f"Validated {len(near)} reviewed nearby and {len(far)} distant additions; total {len(updated['galaxies'])}.")
        if args.dry_run:
            print("No files saved.");return 0
        pending={ROOT/body["raw_record"]:json.dumps(raw,indent=2,allow_nan=False)+"\n" for body,raw in chosen}
        pending.update({ROOT/"data/galaxies.json":json.dumps(updated,indent=2,allow_nan=False)+"\n",
                        ROOT/REPORT:json.dumps(report,indent=2,allow_nan=False)+"\n"})
        write_configs(pending);builder.write_galaxy_configs(configs,updated,ROOT,ids)
        return 0
    except (CatalogError,OSError,KeyError,ValueError,TypeError) as exc:
        print("Cannot add reviewed galaxies: "+str(exc),file=sys.stderr);return 2


if __name__=="__main__":sys.exit(main())
