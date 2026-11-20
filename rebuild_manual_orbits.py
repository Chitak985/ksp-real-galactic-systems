#!/usr/bin/env python3
"""Update hand-written object positions using the shared frozen-orbit adapter."""
import argparse
import math
from pathlib import Path
import re
import sys

import rebuild_galaxies as galaxies
from rebuild_configs import (CatalogError, PARSEC_M, format_number, frozen_orbit,
    heliocentric_position, load_catalog, node_contents, number, orbit_position,
    render_orbit, repository_path, write_configs)

ROOT=Path(__file__).resolve().parent
CATALOG="data/manual_object_positions.json"


def placement(record,settings):
    position=record["position"]
    kind=position["kind"]
    if kind=="preserved_orbit":
        xyz=orbit_position(position["original_orbit"],0)
        return xyz,{"placement_mpc":math.hypot(*xyz)/PARSEC_M/1e6}
    expected_frame="Galactic" if kind=="galactic_distance" else "ICRS"
    if position.get("frame")!=expected_frame or position.get("epoch")!="J2000":
        raise CatalogError("Manual astrometry requires its declared J2000 coordinate frame.")
    if kind=="galactic_distance":
        lon,lat=map(math.radians,(number(position["longitude_deg"],"Galactic longitude"),number(position["latitude_deg"],"Galactic latitude")))
        if not -math.pi/2<=lat<=math.pi/2:raise CatalogError("Invalid Galactic latitude.")
        d=number(position["distance_pc"],"distance_pc",positive=True)*PARSEC_M
        return (d*math.cos(lat)*math.cos(lon),d*math.cos(lat)*math.sin(lon),d*math.sin(lat)),{"placement_mpc":d/PARSEC_M/1e6}
    if kind=="cosmological_redshift":
        distances=galaxies.cosmology(position["redshift"],settings["cosmology"])
        distance_pc=distances["placement_mpc"]*1e6
    elif kind=="icrs_distance":
        distance_pc=number(position["distance_pc"],"distance_pc",positive=True)
        distances={"placement_mpc":distance_pc/1e6,
                   "lookback_years":distance_pc*PARSEC_M/(galaxies.C_KM_S*1000)/galaxies.YEAR_SECONDS}
    else:raise CatalogError("Unsupported manual positioning method: "+str(kind))
    xyz=heliocentric_position({"frame":"ICRS","epoch":"J2000","ra_deg":position["ra_deg"],
                              "dec_deg":position["dec_deg"],"distance_pc":distance_pc})
    return xyz,distances


def span(text,name):
    # Preserve offsets while ignoring comments that could contain braces/names.
    clean=re.sub(r"//[^\r\n]*",lambda m:" "*len(m.group()),text)
    matches=list(re.finditer(r"(?m)^[ \t]*"+re.escape(name)+r"[ \t\r\n]*\{",clean))
    if len(matches)!=1:raise CatalogError("Expected one "+name+" node.")
    start=matches[0].start();opening=clean.index("{",start);depth=1
    for index in range(opening+1,len(clean)):
        depth+=(clean[index]=="{")-(clean[index]=="}")
        if depth==0:return start,index+1,opening
    raise CatalogError("Unclosed "+name+" node.")


def update_config(text,record,settings):
    if record["body_name"].casefold()=="sun":raise CatalogError("Stock Sun must remain the root and cannot be updated.")
    actual=re.findall(r"(?m)^\s*name\s*=\s*([^\r\n/]+)",node_contents(text,"Body"))
    if not actual or actual[0].strip()!=record["body_name"]:raise CatalogError("Manual body identity mismatch.")
    xyz,distances=placement(record,settings)
    if number(settings["frozen_period_seconds"],"frozen period")!=3e300:raise CatalogError("Frozen period must be 3E300.")
    orbit=frozen_orbit(xyz,settings)
    start,end,opening=span(text,"Orbit")
    indent=re.match(r"[ \t]*",text[start:]).group()
    newline="\r\n" if "\r\n" in text else "\n"
    replacement=newline.join(indent+line for line in render_orbit(orbit).splitlines())
    old=text[start:end]
    # Preserve cosmetic camera/color keys without retaining stale orbital data.
    extras=re.findall(r"(?m)^[ \t]*(color|nodeColor|cameraSmaRatioBounds)[ \t]*=[ \t]*([^\r\n]+)",old)
    if extras:
        addition="".join(indent+"    "+key+" = "+value.strip()+newline for key,value in extras)
        replacement=replacement[:-len(indent+"}")]+addition+indent+"}"
    text=text[:start]+replacement+text[end:]
    soi=number(record["core_soi_m"],"core SOI",positive=True)
    if soi>=math.hypot(*xyz):raise CatalogError("Core SOI would enclose Sol.")
    start,end,opening=span(text,"Properties")
    node=text[start:end]
    pattern=r"(?m)^([ \t]*sphereOfInfluence[ \t]*=)[^\r\n]*"
    if len(re.findall(pattern,node))>1:raise CatalogError("Duplicate SOI fields.")
    if re.search(pattern,node):node=re.sub(pattern,lambda m:m.group(1)+" "+format_number(soi),node)
    else:
        local_open=node.index("{")+1
        body_indent=re.match(r"[ \t]*",node).group()+"  "
        node=node[:local_open]+newline+body_indent+"sphereOfInfluence = "+format_number(soi)+node[local_open:]
    return text[:start]+node+text[end:]


def build_updates(catalog,root=ROOT):
    if catalog.get("schema_version")!=1:raise CatalogError("Unsupported manual position schema.")
    outputs={};names=set()
    for record in catalog["objects"]:
        name=record["body_name"].casefold()
        if name in names:raise CatalogError("Duplicate manual body.")
        names.add(name)
        path=repository_path(root,record["path"])
        text=path.read_bytes().decode("utf-8")
        if text.startswith("// Generated by rebuild_"):raise CatalogError("Manual updater cannot edit a generated config.")
        updated=update_config(text,record,catalog["settings"])
        if path in outputs:raise CatalogError("Duplicate manual output path.")
        outputs[path]=updated
    return outputs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    try:
        outputs=build_updates(load_catalog(ROOT/CATALOG),ROOT)
        stale=[path for path,text in outputs.items() if path.read_bytes().decode("utf-8")!=text]
        if args.check:
            for path in stale:print("Needs manual-position update: "+str(path.relative_to(ROOT)))
            if stale:return 1
        elif not args.dry_run:
            # Binary writes preserve each hand-written file's existing newlines.
            import os,uuid
            for path in stale:
                temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
                try:
                    temporary.write_bytes(outputs[path].encode("utf-8"));os.replace(temporary,path)
                finally:
                    if temporary.exists():temporary.unlink()
        print(f"Validated {len(outputs)} manual positions; {len(stale)} configs {'need updating' if args.dry_run else 'updated' if not args.check else 'differ'}.")
        return 0
    except (CatalogError,OSError,KeyError,ValueError,TypeError) as exc:
        print("Cannot update manual positions: "+str(exc),file=sys.stderr);return 2


if __name__=="__main__":sys.exit(main())
