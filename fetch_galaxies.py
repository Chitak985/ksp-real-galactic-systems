#!/usr/bin/env python3
"""Fetch public SIMBAD/VizieR galaxy records, save them, and build KSP configs."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

import rebuild_galaxies as builder
from rebuild_configs import CatalogError, load_catalog, number, write_configs

ROOT = Path(__file__).resolve().parent
SIMBAD = ("https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
          "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync")
VIZIER = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
          "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv")
NEARBY_TABLE = "J/AJ/145/101/catalog"
FIELDS = ("b.main_id,b.ra,b.dec,b.otype,b.morph_type,b.morph_bibcode,"
          "b.galdim_majaxis,b.galdim_minaxis,b.galdim_angle,b.galdim_bibcode,"
          "b.rvz_redshift,b.rvz_radvel,b.rvz_bibcode,f.B,f.V,f.K")


def request(endpoints, params, timeout=12):
    errors = []
    for endpoint in endpoints:
        url = endpoint + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"RealGalacticSystems galaxy importer/1.0"})
            with urllib.request.urlopen(req,timeout=timeout) as response:
                data=response.read(4_000_001)
            if len(data)>4_000_000:
                raise CatalogError("Catalog response exceeds 4 MB.")
            text=data.decode("utf-8-sig")
            if text.lstrip().lower().startswith(("<!doctype html", "<html")):
                raise CatalogError("Catalog returned a web page instead of data.")
            return {"url":url,"text":text}
        except (urllib.error.URLError,TimeoutError,OSError,UnicodeError,CatalogError) as exc:
            errors.append(f"{urllib.parse.urlsplit(endpoint).netloc}: {exc}")
    raise CatalogError("All catalog mirrors failed: "+"; ".join(errors))


def tap(query):
    result=request(SIMBAD,{"request":"doQuery","lang":"adql","format":"json","query":query})
    doc=json.loads(result["text"])
    if not isinstance(doc.get("data"),list) or not isinstance(doc.get("metadata"),list):
        raise CatalogError("SIMBAD returned no valid table.")
    keys=[m["name"] for m in doc["metadata"]]
    result["rows"]=[dict(zip(keys,row)) for row in doc["data"]]
    return result


def parse_tsv(text):
    if "#Table" not in text:
        raise CatalogError("VizieR returned no table.")
    lines=[line for line in text.splitlines() if line and not line.startswith("#")]
    # VizieR omits even the data header when a valid cone query has no matches.
    # Require the expected table and column declarations; do not treat a server
    # error or an unrelated response as an empty nearby-galaxy catalog.
    if not lines:
        columns=set(re.findall(r"(?m)^#Column\t([^\t]+)\t",text))
        failed=re.search(r"(?im)^#(?:ERROR|INFO\s+(?:Error|QUERY_STATUS\s*=\s*ERROR))",text)
        if f"#Name: {NEARBY_TABLE}" in text and {"Name","Dist"} <= columns and not failed:
            return []
    if len(lines)<3:
        raise CatalogError("VizieR table header is incomplete.")
    rows=list(csv.reader(lines,delimiter="\t"))
    if "Name" not in rows[0] or "Dist" not in rows[0]:
        raise CatalogError("Unexpected VizieR nearby-galaxy schema.")
    return [dict(zip(rows[0],(s.strip() for s in row))) for row in rows[3:]
            if len(row)==len(rows[0])]


def slug(name):
    name=re.sub(r"[^a-z0-9]+","_",name.casefold()).strip("_")
    if not re.fullmatch(r"[a-z][a-z0-9_]*",name) or len(name)>80:
        raise CatalogError("Use a short galaxy catalog identifier.")
    return name


def fetch_records(name, simbad_only=False):
    # SQL string literals are escaped; names are never interpreted as code/paths.
    literal=name.replace("'","''")
    result=tap("SELECT TOP 2 "+FIELDS+" FROM basic AS b JOIN ident AS i ON b.oid=i.oidref "
               "LEFT JOIN allfluxes AS f ON b.oid=f.oidref WHERE i.id='"+literal+"'")
    if len(result["rows"])!=1:
        raise CatalogError(f"SIMBAD did not resolve exactly one galaxy for {name!r}.")
    row=result["rows"][0]
    if builder.canonical(row["main_id"]) in builder.EXCLUDED:
        raise CatalogError(f"{name} resolves to an excluded/existing galaxy.")
    if simbad_only:
        # An explicit alternate-source workflow, never an error-to-empty fallback.
        # Missing distances/masses still fail normalize() unless reviewed inputs
        # or valid cosmological data supply them.
        return {"requested_name":name,"retrieved_utc":datetime.now(timezone.utc).isoformat(),
                "simbad":result,"nearby":{"rows":[],"url":None,"text":None,"skipped":True}}
    ra,dec=number(row["ra"],"RA"),number(row["dec"],"Dec")
    nearby=request(VIZIER,{"-source":NEARBY_TABLE,"-c":f"{ra:.10f} {dec:.10f}",
                            "-c.rs":"60","-out.all":"1","-out.max":"20"})
    nearby["rows"]=parse_tsv(nearby["text"])
    if len(nearby["rows"])>1:
        # Do not silently choose a different object in a crowded field.
        raise CatalogError(f"Multiple nearby-catalog matches for {name}; review the records.")
    return {"requested_name":name,"retrieved_utc":datetime.now(timezone.utc).isoformat(),
            "simbad":result,"nearby":nearby}


def optional_float(record,key):
    value=record.get(key)
    if value is None or (isinstance(value,str) and not value.strip()):return None
    return number(float(value),key)


def normalize(raw, settings, overrides=None):
    if raw.get("database")=="LVDB":
        from fetch_nearby_galaxies import normalize_lvdb
        return normalize_lvdb(raw,settings,overrides)
    basic=raw["simbad"]["rows"][0]
    nearby_rows=raw["nearby"]["rows"]
    if len(nearby_rows)>1:
        raise CatalogError("Ambiguous nearby galaxy match.")
    nearby=nearby_rows[0] if nearby_rows else {}
    name=" ".join(basic["main_id"].split())
    if builder.canonical(name) in builder.EXCLUDED:
        raise CatalogError(f"Excluded galaxy: {name}")
    galaxy={"id":slug(raw["requested_name"]),"body_name":re.sub(r"[^A-Za-z0-9_]","",name),
            "name":name,"ra_deg":basic["ra"],"dec_deg":basic["dec"]}
    # Missing values require a reviewed override instead of a fabricated measurement.
    redshift=optional_float(basic,"rvz_redshift")
    if redshift is None:
        velocity=optional_float(nearby,"HRV")
        redshift=None if velocity is None else velocity/builder.C_KM_S
    galaxy["redshift"]=redshift
    distance=optional_float(nearby,"Dist")
    if distance is not None:
        galaxy["distance"]={"kind":"redshift_independent","mpc":distance,
                            "method":nearby.get("f_Dist"),"source":"2013AJ....145..101K"}
    elif redshift is not None and redshift>=0.01:
        galaxy["distance"]={"kind":"cosmological_redshift","cosmological_redshift":redshift,
                            "method":"flat FLRW; assumes observed redshift is cosmological"}
    else:
        galaxy["distance"]={"kind":"redshift_independent","mpc":None,"method":"REVIEW REQUIRED"}
    major=optional_float(nearby,"a26")
    ratio=optional_float(nearby,"b/a")
    if major is None:
        major=optional_float(basic,"galdim_majaxis")
        minor=optional_float(basic,"galdim_minaxis")
        ratio=minor/major if major and minor else None
    galaxy.update(angular_major_arcmin=major,axis_ratio=ratio,
                  position_angle_deg=optional_float(basic,"galdim_angle"),
                  morphology=basic.get("morph_type") or nearby.get("Mcl") or str(nearby.get("TT","")),
                  hubble_t=optional_float(nearby,"TT"),
                  b_magnitude=optional_float(nearby,"Bmag") if nearby else optional_float(basic,"B"))
    override=copy.deepcopy(overrides or {})
    allowed={"template","body_name","name","distance","angular_major_arcmin","axis_ratio",
             "position_angle_deg","morphology","hubble_t","mass_solar","mass_kind","b_magnitude","redshift","uncertainty_note","far"}
    if set(override)-allowed:
        raise CatalogError("Unknown override fields: "+", ".join(sorted(set(override)-allowed)))
    galaxy.update(override)
    if "template" not in galaxy:
        galaxy["template"]=builder.classify(galaxy["morphology"],nearby.get("Mcl",""),optional_float(nearby,"TT"))
    # A round, featureless object has an unconstrained PA; using zero is explicit.
    if galaxy["position_angle_deg"] is None and galaxy["axis_ratio"] is not None and galaxy["axis_ratio"]>=0.98:
        galaxy["position_angle_deg"]=0
    if "mass_solar" not in galaxy:
        logmass=optional_float(nearby,"M26")
        if logmass is not None:
            galaxy["mass_solar"]=10**logmass
            galaxy["mass_kind"]="indicative mass within Holmberg radius; point-mass game approximation"
        else:
            loglum=optional_float(nearby,"KLum")
            if loglum is None:
                k=optional_float(basic,"K")
                if k is not None:
                    dl=builder.distance_measures(galaxy,settings)["luminosity_mpc"]*1e6
                    # Solar absolute Ks magnitude and M/L=1 are explicit estimates.
                    loglum=-0.4*(k-5*math.log10(dl/10)-3.28)
            galaxy["mass_solar"]=10**loglum if loglum is not None else None
            galaxy["mass_kind"]="estimated stellar mass: Ks mass-to-light ratio 1 solar; excludes dark halo"
    galaxy.setdefault("uncertainty_note",
        "Stellar mass is estimated from Ks light; distance, 3D depth, colors and core gravity are uncertain."
        if galaxy["mass_kind"].startswith("estimated") else
        "Distances have catalog uncertainties; 3D depth, colors and core gravity are approximate.")
    galaxy["sources"]=[
        {"database":"SIMBAD","url":raw["simbad"]["url"],"retrieved_utc":raw["retrieved_utc"],
         "bibcodes":{k:basic.get(k) for k in ("morph_bibcode","galdim_bibcode","rvz_bibcode")},
         "fields":["ra_deg","dec_deg","morphology","position_angle_deg","redshift"]},
        {"database":"VizieR Updated Nearby Galaxy Catalog","catalog":NEARBY_TABLE,
         "url":raw["nearby"]["url"],"bibcode":"2013AJ....145..101K",
         "fields":["distance","angular_major_arcmin","axis_ratio","mass_solar","b_magnitude"] if nearby else [],
         "matched_name":nearby.get("Name")},
    ]
    galaxy["raw_record"]=f"data/galaxy_sources/{slug(raw['requested_name'])}.json"
    galaxy["adoption_notes"]={
        "geometry":"Observed angular Holmberg diameter and axial ratio; SIMBAD PA; disk inclination assumes intrinsic flattening.",
        "lookback":"Observed morphology is already retarded; no additional rewinding/evolution is applied.",
        "color":"Template blackbody-equivalent palettes shifted by observed redshift; approximate, not measured spectra.",
        "gravity":galaxy["mass_kind"],
        "overrides":override,
    }
    if not nearby:
        galaxy["sources"][0]["fields"] += ["angular_major_arcmin","axis_ratio","b_magnitude"]
        galaxy["sources"][1]["note"]="No nearby-catalog match; no measurements adopted from this query."
        galaxy["adoption_notes"]["geometry"]="SIMBAD isophotal dimensions or reviewed deep-starlight extent; bandpass and depth vary."
    if raw["nearby"].get("skipped"):
        if nearby:
            raise CatalogError("A skipped nearby query cannot contain catalog rows.")
        galaxy["sources"]=galaxy["sources"][:1]
        galaxy["adoption_notes"]["nearby_catalog"]="Explicitly not queried; no nearby-catalog measurements adopted."
    builder.replacements(galaxy,settings)
    return galaxy


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names",nargs="+",help='Catalog names, e.g. "NGC 300" "NGC 404".')
    parser.add_argument("--catalog",type=Path,default=ROOT/"data/galaxies.json")
    parser.add_argument("--overrides",type=Path,help="JSON mapping requested names to reviewed replacement fields.")
    parser.add_argument("--refresh",action="store_true",help="Explicitly replace existing saved records and their adopted values.")
    parser.add_argument("--offline",action="store_true",help="Use saved raw records; make no network requests.")
    parser.add_argument("--simbad-only",action="store_true",help="Explicitly omit the nearby-catalog query; reviewed replacement data may be needed.")
    parser.add_argument("--catalog-only",action="store_true",help="Save fetched data without writing game configs.")
    parser.add_argument("--dry-run",action="store_true",help="Fetch/validate in memory without saving any files.")
    args=parser.parse_args(argv)
    try:
        catalog=load_catalog(args.catalog) if args.catalog.exists() else builder.default_catalog()
        overrides=load_catalog(args.overrides) if args.overrides else {}
        by_id={g["id"]:g for g in catalog["galaxies"]}
        if len(by_id)!=len(catalog["galaxies"]):
            raise CatalogError("Existing catalog has duplicate galaxy IDs.")
        pending={}
        for name in args.names:
            identifier=slug(name)
            if identifier in by_id and not args.refresh:
                print(f"Kept saved {identifier}; use --refresh to fetch replacement values.")
                continue
            cache=ROOT/f"data/galaxy_sources/{identifier}.json"
            print(f"{'Reading saved records for' if args.offline else 'Fetching'} {name}...")
            raw=load_catalog(cache) if args.offline else fetch_records(name,simbad_only=args.simbad_only)
            if slug(raw["requested_name"])!=identifier:
                raise CatalogError("Raw cache name does not match the requested galaxy.")
            normalized=normalize(raw,catalog["settings"],overrides.get(name))
            normalized.setdefault("far",by_id.get(identifier,{}).get("far","None"))
            by_id[identifier]=normalized
            pending[cache]=json.dumps(raw,indent=2,ensure_ascii=True,allow_nan=False)+"\n"
        catalog["galaxies"]=sorted(by_id.values(),key=lambda g:builder.distance_measures(g,catalog["settings"])["placement_mpc"])
        configs=builder.build_configs(catalog,ROOT)
        if args.dry_run:
            print(f"Validated {len(configs)} galaxy config(s); no files saved.")
            return 0
        # Validate the entire batch before writing caches, adopted catalog, or configs.
        pending[args.catalog]=json.dumps(catalog,indent=2,ensure_ascii=True,allow_nan=False)+"\n"
        write_configs(pending)
        if not args.catalog_only:builder.write_galaxy_configs(configs,catalog,ROOT)
        print(f"Saved {len(catalog['galaxies'])} galaxy record(s)"+
              ("." if args.catalog_only else f" and rebuilt {len(configs)} config(s)."))
        return 0
    except (CatalogError,OSError,KeyError,TypeError,ValueError) as exc:
        print(f"Cannot import galaxies: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    sys.exit(main())
