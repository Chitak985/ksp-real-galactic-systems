#!/usr/bin/env python3
"""Add the nearest confirmed LVDB spheroidal galaxies from versioned measurements."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import urllib.request

import rebuild_galaxies as builder
from rebuild_configs import CatalogError, config_text, load_catalog, number, write_configs

ROOT=Path(__file__).resolve().parent
FAMILIES={"dSph":"dwarf_spheroidal","dE":"dwarf_elliptical",
          "dE/dSph":"dwarf_elliptical","cE":"compact_elliptical"}
LY_PER_KPC=3261.563777167433


def canonical(value):
    """Compare catalog aliases without padding zeros (NGC 0185 == NGC185)."""
    return re.sub(r"(?<=[a-z])0+(?=\d)","",builder.canonical(value))


def scalar(row,key,positive=False):
    value=row.get(key)
    if value is None or str(value).strip()=="":
        raise CatalogError(f"Missing LVDB measurement: {key}")
    try:value=float(value)
    except (TypeError,ValueError) as exc:raise CatalogError(f"Invalid LVDB measurement: {key}") from exc
    return number(value,"LVDB "+key,positive=positive)


def snapshot_rows(snapshot,review):
    if snapshot.get("version")!=review["version"]:
        raise CatalogError("LVDB snapshot and reviewed metadata versions differ.")
    payload=snapshot["text"].encode("utf-8")
    digest=hashlib.sha256(payload).hexdigest()
    if digest!=review["csv_sha256"] or digest!=snapshot.get("csv_sha256"):
        raise CatalogError("LVDB snapshot checksum does not match the published release.")
    reader=csv.DictReader(io.StringIO(snapshot["text"]))
    required={"key","ra","dec","distance","rhalf","confirmed_galaxy","confirmed_real","table"}
    if not required <= set(reader.fieldnames or []):
        raise CatalogError("Unexpected LVDB CSV schema.")
    rows=list(reader)
    if len({row["key"] for row in rows})!=len(rows):
        raise CatalogError("Duplicate LVDB keys.")
    return rows


def normalize_lvdb(raw,settings,overrides=None):
    """Keep measured units explicit; derive a Gaussian display from half-light size."""
    if raw.get("database")!="LVDB" or raw.get("schema_version")!=1:
        raise CatalogError("Unsupported LVDB raw-record schema.")
    if overrides:
        raise CatalogError("LVDB corrections belong in the reviewed LVDB metadata, not SIMBAD overrides.")
    row,meta=copy.deepcopy(raw["row"]),raw["metadata"]
    literature=raw.get("literature_review")
    if literature is not None:
        if literature.get("schema_version")!=1 or not literature.get("sources"):
            raise CatalogError("A literature review requires its schema and primary sources.")
        allowed={"ra","dec","distance","distance_em","distance_ep","distance_measurement_method",
                 "rhalf","ellipticity","position_angle","apparent_magnitude_v","mass_stellar",
                 "ref_distance","ref_structure","ref_m_v"}
        updates=literature.get("row_updates",{})
        if set(updates)-allowed:
            raise CatalogError("Unsupported literature measurement update.")
        row.update(updates)
    key=row["key"]
    if not re.fullmatch(r"[a-z][a-z0-9_]*",key) or meta["key"]!=key:
        raise CatalogError("LVDB key is unsafe or does not match its metadata.")
    if row.get("confirmed_galaxy")!="1" or row.get("confirmed_real")!="1" or row.get("confirmed_star_cluster")=="1":
        raise CatalogError("LVDB does not identify this object as a confirmed galaxy.")
    if not row.get("table","").startswith("dwarf_"):
        raise CatalogError("This LVDB table is not a dwarf-galaxy catalog.")
    morphology=meta["type"]
    if morphology not in FAMILIES:
        raise CatalogError("Unsupported nearby morphology: "+str(morphology))
    family=FAMILIES[morphology]
    rh=scalar(row,"rhalf",positive=True)  # Projected SEMI-MAJOR half-light radius, arcmin.
    distance_kpc=scalar(row,"distance",positive=True)  # Heliocentric, not Galactocentric.
    extent=number(raw["assumptions"]["extent_sigma"],"Gaussian extent",positive=True)
    if not 2 <= extent <= 6 or rh>=1800:
        raise CatalogError("Invalid nearby Gaussian extent or half-light angle.")
    # For a projected Gaussian, r_half = sigma * sqrt(2 ln 2). Use tangents
    # rather than a small-angle shortcut for unusually broad nearby dwarfs.
    diameter=math.degrees(2*math.atan(extent*math.tan(math.radians(rh/60))/math.sqrt(2*math.log(2))))*60
    round_assumed=False
    if not row.get("ellipticity"):
        upper=scalar(row,"ellipticity_ul")
        if not 0 <= upper <= .1:
            raise CatalogError("No measured flattening or tight near-circular upper limit.")
        ellipticity=0.0;round_assumed=True
    else:ellipticity=scalar(row,"ellipticity")
    if not 0 <= ellipticity < 1:
        raise CatalogError("LVDB ellipticity must be in [0,1).")
    if not row.get("position_angle"):
        if ellipticity>.02:
            raise CatalogError("Non-round galaxy has no sky position angle.")
        pa=0.0
    else:pa=scalar(row,"position_angle")%180
    logmass=scalar(row,"mass_stellar")  # LOG10 Msun, not a linear mass.
    if not -3 <= logmass <= 15:
        raise CatalogError("LVDB stellar mass is outside the supported range.")
    velocity_status="measured"
    velocity_review=literature.get("radial_velocity",{}) if literature else {}
    if row.get("vlos_systemic"):
        if velocity_review:
            raise CatalogError("Reviewed velocity policy cannot discard a catalog measurement.")
        velocity=scalar(row,"vlos_systemic")  # Heliocentric km/s.
    elif velocity_review.get("status")=="unmeasured":
        velocity=None;velocity_status="unmeasured"
        if velocity_review.get("km_s") is not None:
            raise CatalogError("An unknown radial velocity must remain null.")
    elif velocity_review.get("status") in {"cluster_proxy","model_proxy"}:
        velocity_status=velocity_review["status"]
        velocity=number(velocity_review["km_s"],"reviewed velocity proxy")
        if not velocity_review.get("note"):
            raise CatalogError("Velocity proxies require an uncertainty note.")
    else:
        velocity=scalar(row,"vlos_systemic")
    name=config_text(row["name"],"LVDB name")
    body="".join(part[:1].upper()+part[1:] for part in key.split("_"))
    notes=("Gaussian extent is inferred from measured half-light size; profile, depth and stellar M/L are estimates. "
           "V brightness is extinction-corrected; foreground dimming and tidal debris are not reconstructed.")
    if round_assumed:notes+=" Near-circular shape is assumed from an ellipticity upper limit."
    special=raw["assumptions"].get("notes","")
    if special:notes+=" "+config_text(special,"nearby uncertainty note")
    if literature:
        notes+=" "+config_text(literature["note"],"literature uncertainty note")
        if velocity_review:notes+=" "+config_text(velocity_review["note"],"velocity uncertainty note")
    entry={
        "id":key,"body_name":body,"name":name,"template":family,"lvdb_key":key,
        "ra_deg":scalar(row,"ra"),"dec_deg":scalar(row,"dec"),
        "redshift":velocity/builder.C_KM_S if velocity is not None else None,
        "distance":{"kind":"redshift_independent","mpc":distance_kpc/1000,
                    "uncertainty_minus_mpc":scalar(row,"distance_em")/1000 if row.get("distance_em") else None,
                    "uncertainty_plus_mpc":scalar(row,"distance_ep")/1000 if row.get("distance_ep") else None,
                    "method":row.get("distance_measurement_method") or "published stellar distance modulus",
                    "source":row.get("ref_distance")},
        "angular_major_arcmin":diameter,"axis_ratio":1-ellipticity,"position_angle_deg":pa,
        "morphology":morphology,"hubble_t":None,"b_magnitude":None,
        "v_magnitude":scalar(row,"apparent_magnitude_v"),"photometry_extinction_corrected":True,
        "mass_solar":10**logmass,
        "mass_kind":"estimated stellar mass: LVDB log10 mass, assumes V-band M/L=2 solar; excludes dark halo; game point-mass proxy",
        "spheroid_extent_sigma":extent,"half_light_major_arcmin":rh,"uncertainty_note":notes,
        "sources":[{"database":"LVDB","version":raw["version"],"url":raw["csv_url"],
                    "sha256":raw["csv_sha256"],"retrieved_utc":raw["retrieved_utc"],
                    "metadata_url":raw["metadata_url"],
                    "references":{k:v for k,v in row.items() if k.startswith("ref_") and v},
                    "fields":["ra_deg","dec_deg","distance","half_light_major_arcmin","axis_ratio",
                              "position_angle_deg","v_magnitude","mass_solar","morphology"]}],
        "raw_record":f"data/galaxy_sources/lvdb_{key}.json",
        "adoption_notes":{"geometry":"Measured elliptical half-light semi-major axis mapped to a Gaussian; nominal extent is a model boundary, not a measured tidal/isophotal diameter.",
                          "photometry":"Published extinction-corrected V magnitude; no invented B magnitude or B-V color.",
                          "gravity":"Stellar M/L=2 estimate, not a dynamical or central black-hole mass.",
                          "redshift":"Local v/c Doppler proxy from heliocentric radial velocity; not a cosmological distance clock.",
                          "lookback":"Observed snapshot; approximate D/c lookback, no second structural evolution.",
                          "overrides":{}},
    }
    if entry["distance"]["method"] in {"host","nam","tf","btf"}:
        entry["distance"]["kind"]="adopted_local"
    if literature:
        entry["redshift_status"]=velocity_status
        entry["radial_velocity_review"]=copy.deepcopy(velocity_review)
        entry["sources"].append({"database":"Reviewed primary literature","reviewed_utc":literature["reviewed_utc"],
                                 "sources":literature["sources"],"fields":list(literature.get("row_updates",{})),
                                 "radial_velocity":copy.deepcopy(velocity_review)})
        entry["adoption_notes"]["literature_review"]=copy.deepcopy(literature)
        if velocity_status!="measured":
            entry["adoption_notes"]["redshift"]=("No measured galaxy systemic radial velocity. "
                "Unknown values remain null with an unshifted display palette; explicit proxies are labeled. Local distance and lookback do not use this value.")
        if literature.get("mass_kind"):
            entry["mass_kind"]=config_text(literature["mass_kind"],"reviewed mass kind")
            entry["adoption_notes"]["gravity"]=entry["mass_kind"]
    builder.replacements(entry,settings)
    return entry


def select_nearest(rows,review,snapshot,catalog,count):
    existing=set(builder.EXCLUDED)|{"mw","lmc","smc"}
    for body in catalog["galaxies"]:
        existing.update(canonical(body[k]) for k in ("id","name","body_name"))
    selected=[];skipped=[]
    ordered=sorted((row for row in rows if row.get("distance") and row["key"] in review["metadata"]),
                   key=lambda row:(scalar(row,"distance",positive=True),row["key"]))
    for row in ordered:
        key=row["key"];meta=review["metadata"].get(key)
        if meta is None:continue
        try:
            aliases=[key,row["name"],*meta.get("aliases",[])]
            if any(canonical(alias) in existing for alias in aliases):
                raise CatalogError("Already present or excluded.")
            if key in review["excluded"]:
                raise CatalogError(review["excluded"][key]["reason"])
            raw={"schema_version":1,"database":"LVDB","version":snapshot["version"],
                 "csv_url":snapshot["csv_url"],"csv_sha256":snapshot["csv_sha256"],
                 "retrieved_utc":snapshot["retrieved_utc"],"row":row,"metadata":meta,
                 "metadata_url":review["metadata_base_url"]+key+".yaml",
                 "assumptions":{"extent_sigma":review["extent_sigma"],
                                "notes":review.get("uncertainty_notes",{}).get(key,"")}}
            body=normalize_lvdb(raw,catalog["settings"])
        except (CatalogError,KeyError,ValueError) as exc:
            skipped.append({"key":key,"distance_kpc":scalar(row,"distance"),"reason":str(exc)})
            continue
        # Distance is primary; every object closer than this one was considered.
        selected.append((body,raw))
        existing.update(canonical(alias) for alias in [body["id"],body["name"],body["body_name"],*meta.get("aliases",[])])
        if len(selected)==count:break
    if len(selected)<count:
        raise CatalogError(f"Only {len(selected)} eligible new galaxies have sufficiently complete data; requested {count}.")
    return selected,skipped


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count",type=int,default=50,help="Number of nearest NEW eligible galaxies to add.")
    parser.add_argument("--offline",action="store_true",help="Use the saved, checksum-verified source snapshot.")
    parser.add_argument("--dry-run",action="store_true",help="Validate and list additions without writing any files.")
    args=parser.parse_args(argv)
    try:
        if not 1 <= args.count <= 1000:raise CatalogError("count must be between 1 and 1000.")
        review=load_catalog(ROOT/"data/nearby_galaxy_review.json")
        snapshot_path=ROOT/"data/galaxy_sources/lvdb_nearby_snapshot.json"
        if args.offline:snapshot=load_catalog(snapshot_path)
        else:
            request=urllib.request.Request(review["csv_url"],headers={"User-Agent":"RealGalacticSystems nearby importer/1.0"})
            with urllib.request.urlopen(request,timeout=30) as response:payload=response.read(4_000_001)
            if len(payload)>4_000_000:raise CatalogError("LVDB source exceeds 4 MB.")
            snapshot={"database":"LVDB","version":review["version"],"csv_url":review["csv_url"],
                      "csv_sha256":hashlib.sha256(payload).hexdigest(),
                      "retrieved_utc":datetime.now(timezone.utc).isoformat(),"text":payload.decode("utf-8")}
        rows=snapshot_rows(snapshot,review)
        catalog=load_catalog(ROOT/"data/galaxies.json")
        chosen,skipped=select_nearest(rows,review,snapshot,catalog,args.count)
        updated=copy.deepcopy(catalog)
        updated["galaxies"].extend(body for body,raw in chosen)
        updated["galaxies"].sort(key=lambda body:builder.distance_measures(body,updated["settings"])["placement_mpc"])
        reviewed=updated.setdefault("additional_reviewed_adoptions_files",[])
        if "data/nearby_galaxy_review.json" not in reviewed:reviewed.append("data/nearby_galaxy_review.json")
        identifiers=[body["id"] for body,raw in chosen]
        configs=builder.build_configs(updated,ROOT,identifiers)
        report={"version":snapshot["version"],"selection":"Nearest eligible confirmed galaxies, ordered by nominal heliocentric distance.",
                "selected_ids":identifiers,"within_100000_ly":sum(body["distance"]["mpc"]*1000*LY_PER_KPC<=100000 for body,raw in chosen),
                "skipped_before_last_addition":skipped}
        report_path=ROOT/"data/nearby_galaxy_selection.json"
        if report_path.exists():
            previous=load_catalog(report_path)
            history=previous.pop("previous_selections",[])
            report["previous_selections"]=[*history,previous]
        for body,raw in chosen:print(f"{body['name']}: {body['distance']['mpc']*1000*LY_PER_KPC:,.0f} light-years")
        if args.dry_run:
            print(f"Validated {len(configs)} additions; no files saved.");return 0
        pending={ROOT/body["raw_record"]:json.dumps(raw,indent=2,allow_nan=False)+"\n" for body,raw in chosen}
        pending.update({ROOT/"data/galaxies.json":json.dumps(updated,indent=2,allow_nan=False)+"\n",
                        snapshot_path:json.dumps(snapshot,indent=2,allow_nan=False)+"\n",
                        report_path:json.dumps(report,indent=2,allow_nan=False)+"\n"})
        write_configs(pending);write_configs(configs)
        print(f"Added {len(configs)} nearby galaxies; catalog now contains {len(updated['galaxies'])}.")
        return 0
    except (CatalogError,OSError,KeyError,TypeError,ValueError) as exc:
        print("Cannot import nearby galaxies: "+str(exc),file=sys.stderr);return 2


if __name__=="__main__":sys.exit(main())
