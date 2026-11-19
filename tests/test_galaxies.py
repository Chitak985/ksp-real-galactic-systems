"""Offline regressions for catalog imports, galaxy geometry, and template builds."""

import copy
import io
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock
import urllib.error

import fetch_galaxies as fetch
import rebuild_galaxies as galaxy
from rebuild_configs import CatalogError, PARSEC_M, load_catalog, node_contents, orbit_position, strip_comments, write_configs

ROOT=Path(__file__).resolve().parents[1]


def multiply(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def unity_matrix(angles):
    x,y,z=map(math.radians,angles)
    cx,sx,cy,sy,cz,sz=math.cos(x),math.sin(x),math.cos(y),math.sin(y),math.cos(z),math.sin(z)
    # Independent forward construction of Unity's documented Z-X-Y convention.
    rx=[[1,0,0],[0,cx,-sx],[0,sx,cx]]
    ry=[[cy,0,sy],[0,1,0],[-sy,0,cy]]
    rz=[[cz,-sz,0],[sz,cz,0],[0,0,1]]
    return multiply(ry,multiply(rx,rz))


def dot(a,b):return sum(x*y for x,y in zip(a,b))


class GalaxyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.full_catalog=load_catalog(ROOT/"data/galaxies.json")
        # Keep the original nearby sample as the local-distance regression fixture.
        # The complete, growing catalog is checked separately below.
        self.catalog=copy.deepcopy(self.full_catalog)
        nearby_ids={"ngc_147","ngc_185","ngc_300","ngc_404","ngc_2403"}
        self.catalog["galaxies"]=[b for b in self.catalog["galaxies"] if b["id"] in nearby_ids]
        self.catalog["rendering_overrides"]={k:v for k,v in self.catalog.get("rendering_overrides",{}).items() if k in nearby_ids}
        shutil.copytree(ROOT/"templates/galaxies",self.root/"templates/galaxies")

    def test_five_galaxies_flat_layout_barycenters_and_core_soi(self):
        configs=galaxy.build_configs(self.catalog,self.root)
        self.assertEqual({p.stem for p in configs},{"ngc_147","ngc_185","ngc_300","ngc_404","ngc_2403"})
        for body in self.catalog["galaxies"]:
            path=self.root/f"RealGalacticSystems/Systems/{body['id']}.cfg"
            text=configs[path]
            values=galaxy.replacements(body,self.catalog["settings"])
            self.assertEqual(path.parent,self.root/"RealGalacticSystems/Systems")
            self.assertIn("name = Jool",text)
            self.assertIn("invisible = true",text)
            self.assertIn("radius = 1",text)
            self.assertIn("renderMode = Volumetric",text)
            self.assertIn("alignRotationToOrbit = false",text)
            self.assertIn("rotationPeriod = 0",text)
            self.assertIn("period = 3E300",text)
            self.assertIn("referenceBody = Sun",text)
            self.assertIn("mode = OFF",text)
            self.assertNotIn("{{",text)
            self.assertNotIn("...",strip_comments(text))
            self.assertLessEqual(float(values["soi"])/PARSEC_M,100.0000001)
            mass=float(values["mass_kg"])
            self.assertAlmostEqual(float(values["gravity_g"])*galaxy.STANDARD_GRAVITY/(galaxy.G*mass),1,places=13)

    def test_complete_saved_catalog_includes_giants_with_frozen_placement(self):
        configs=galaxy.build_configs(self.full_catalog,self.root)
        self.assertEqual(len(configs),len(self.full_catalog["galaxies"]))
        giant_ids={"ic_1101","ugc_1382","ugc_2885","ngc_4889","ngc_6166"}
        self.assertTrue(giant_ids <= {b["id"] for b in self.full_catalog["galaxies"]})
        for b in self.full_catalog["galaxies"]:
            text=configs[self.root/f"RealGalacticSystems/Systems/{b['id']}.cfg"]
            d=galaxy.distance_measures(b,self.full_catalog["settings"])
            self.assertGreater(d["lookback_years"],0)
            self.assertIn("referenceBody = Sun",text)
            self.assertIn("period = 3E300",text)
            self.assertIn("invisible = true",text)
            if b["id"] in giant_ids:
                diameter=2*d["angular_diameter_mpc"]*1000*math.tan(math.radians(b["angular_major_arcmin"]/120))
                self.assertGreater(diameter,70)  # kpc: genuinely large stellar bodies
                self.assertLess(d["lookback_years"],2e9)
                self.assertIn("estimate",b["uncertainty_note"].lower()+" "+b["mass_kind"].lower())

    def test_full_catalog_raw_records_and_reviewed_adoptions_replay(self):
        for b in self.full_catalog["galaxies"]:
            raw=load_catalog(ROOT/b["raw_record"])
            with mock.patch.object(fetch.urllib.request,"urlopen",side_effect=AssertionError("network used")):
                adopted=fetch.normalize(raw,self.full_catalog["settings"],b["adoption_notes"]["overrides"])
            self.assertEqual(adopted,b)

    def test_empty_nearby_catalog_is_valid_but_server_errors_are_not(self):
        raw=load_catalog(ROOT/"data/galaxy_sources/ic_1101.json")
        self.assertEqual(fetch.parse_tsv(raw["nearby"]["text"]),[])
        with self.assertRaises(CatalogError):
            fetch.parse_tsv(raw["nearby"]["text"]+"\n#INFO\tError\tquery failed\n")
        with self.assertRaises(CatalogError):
            fetch.parse_tsv("#Table\tunknown\n#Name: wrong/catalog\n")

    def test_explicit_simbad_only_import_keeps_provenance_and_requires_distance(self):
        original=load_catalog(ROOT/"data/galaxy_sources/ngc_300.json")
        with mock.patch.object(fetch,"tap",return_value=original["simbad"]), \
             mock.patch.object(fetch,"request",side_effect=AssertionError("nearby query made")):
            raw=fetch.fetch_records("NGC 300",simbad_only=True)
        self.assertTrue(raw["nearby"]["skipped"])
        self.assertEqual(raw["nearby"]["rows"],[])
        with self.assertRaises(CatalogError):
            fetch.normalize(raw,self.catalog["settings"])
        adopted=fetch.normalize(raw,self.catalog["settings"],{
            "distance":{"kind":"adopted_local","mpc":2.15,"method":"reviewed local compilation"}})
        self.assertEqual(len(adopted["sources"]),1)
        self.assertEqual(adopted["sources"][0]["database"],"SIMBAD")
        self.assertIn("Explicitly not queried",adopted["adoption_notes"]["nearby_catalog"])

    def test_simbad_only_marker_cannot_hide_nearby_measurements(self):
        raw=load_catalog(ROOT/"data/galaxy_sources/ngc_300.json")
        raw["nearby"]["skipped"]=True
        with self.assertRaises(CatalogError):fetch.normalize(raw,self.catalog["settings"])

    def test_adopted_local_distance_is_not_a_cosmological_redshift_clock(self):
        body=copy.deepcopy(self.catalog["galaxies"][0])
        body["distance"]={"kind":"adopted_local","mpc":12.3,"method":"group/flow compilation"}
        body["redshift"]=-0.001
        actual=galaxy.distance_measures(body,self.catalog["settings"])
        self.assertEqual(actual["placement_mpc"],12.3)
        self.assertEqual(actual["angular_diameter_mpc"],12.3)
        self.assertAlmostEqual(actual["lookback_years"]/(12.3e6*3.261563777),1,places=8)
        for value in [None,0,-1,float("nan")]:
            body["distance"]["mpc"]=value
            with self.assertRaises(CatalogError):galaxy.distance_measures(body,self.catalog["settings"])

    def test_spiral_batch_replays_survey_angles_and_mass_units(self):
        adopted=load_catalog(ROOT/"data/spiral_galaxy_adoptions.json")
        ids={fetch.slug(name) for name in adopted if not name.startswith("_")}
        bodies={b["id"]:b for b in self.full_catalog["galaxies"] if b["id"] in ids}
        self.assertEqual(len(bodies),30)
        self.assertTrue(all(b["template"] in {"spiral","weakly_barred_spiral","barred_spiral"} for b in bodies.values()))
        for identifier,body in bodies.items():
            refinement=self.full_catalog["rendering_overrides"][identifier]
            for source in refinement["sources"]:
                measured=source.get("reported_values",{})
                if "inclination_deg" in measured:
                    self.assertEqual(refinement["inclination_deg"],measured["inclination_deg"])
                if "position_angle_deg" in measured:
                    self.assertAlmostEqual(refinement["position_angle_deg"],measured["position_angle_deg"]%180)
        sparc=load_catalog(ROOT/"data/galaxy_sources/spiral_batch_sparc.json")
        row=next(row for row in sparc["rows"] if row["name"]=="UGC00128")
        self.assertEqual(bodies["ugc_128"]["mass_solar"],.5*row["luminosity_3_6_solar"])

    def test_epoch_position_recovers_observed_sky_coordinates(self):
        for body in self.catalog["galaxies"]:
            v=galaxy.replacements(body,self.catalog["settings"])
            orbit={}
            for line in v["orbit"].splitlines():
                if "=" in line:
                    k,value=map(str.strip,line.split("=",1))
                    orbit[k]=value if k in {"referenceBody","mode"} else float(value)
            point=orbit_position(orbit)
            expected=galaxy.heliocentric_position({"frame":"ICRS","epoch":"J2000","ra_deg":body["ra_deg"],
                "dec_deg":body["dec_deg"],"distance_pc":body["distance"]["mpc"]*1e6})
            self.assertLess(math.dist(point,expected)/math.hypot(*expected),1e-14)
            self.assertEqual(orbit_position(orbit,1e15),point)

    def test_projected_major_axis_and_disk_inclination(self):
        # Exercise sky poles, PA wrapping, face-on and edge-on disks.
        for ra,dec in [(0,0),(114.21390893,65.60268078),(13.72,-37.68),(0,90),(180,-90)]:
            for pa in [-20,0,35,123,179]:
                for inc in [0,25,60,90]:
                    with self.subTest(ra=ra,dec=dec,pa=pa,inc=inc):
                        r=unity_matrix(galaxy.orientation(ra,dec,pa,inc))
                        major=tuple(r[i][0] for i in range(3));normal=tuple(r[i][1] for i in range(3))
                        radial,east,north=map(galaxy.icrs_to_unity,galaxy.sky_basis(ra,dec))
                        actual_pa=math.degrees(math.atan2(dot(major,east),dot(major,north)))
                        self.assertAlmostEqual(math.sin(math.radians(actual_pa-pa)),0,places=10)
                        self.assertAlmostEqual(abs(dot(normal,radial)),math.cos(math.radians(inc)),places=10)
                        self.assertAlmostEqual(dot(major,normal),0,places=10)

    def test_observed_angular_size_is_preserved(self):
        for b in self.catalog["galaxies"]:
            v=galaxy.replacements(b,self.catalog["settings"])
            radius=float(v["object_scale"])*6000*100
            distance=b["distance"]["mpc"]*1e6*PARSEC_M
            diameter=math.degrees(2*math.atan(radius/distance))*60
            self.assertAlmostEqual(diameter,b["angular_major_arcmin"],places=10)

    def test_local_blueshift_does_not_produce_negative_distance_or_age(self):
        b=next(b for b in self.catalog["galaxies"] if b["id"]=="ngc_185")
        self.assertLess(b["redshift"],0)
        d=galaxy.distance_measures(b,self.catalog["settings"])
        self.assertEqual(d["placement_mpc"],0.61)
        self.assertAlmostEqual(d["lookback_years"]/(0.61e6*3.261563777),1,places=8)

    def test_cosmology_distances_and_lookback_against_matter_dominated_limit(self):
        # Near Einstein-de Sitter has a closed-form independent solution.
        params={"h0_km_s_mpc":70,"omega_m":0.99999999,"omega_r":0}
        d=galaxy.cosmology(1,params)
        expected=2*galaxy.C_KM_S/70*(1-1/math.sqrt(2))
        self.assertAlmostEqual(d["placement_mpc"]/expected,1,places=7)
        expected_t=2/3*(1e3*PARSEC_M/70)*(1-2**-1.5)/galaxy.YEAR_SECONDS
        self.assertAlmostEqual(d["lookback_years"]/expected_t,1,places=7)
        self.assertEqual(d["luminosity_mpc"],4*d["angular_diameter_mpc"])

    def test_cosmological_placement_retains_observed_angular_size(self):
        b=copy.deepcopy(self.catalog["galaxies"][-1])
        b["redshift"]=1;b["distance"]={"kind":"cosmological_redshift","cosmological_redshift":1}
        v=galaxy.replacements(b,self.catalog["settings"])
        d=galaxy.distance_measures(b,self.catalog["settings"])
        radius=float(v["object_scale"])*6000*100/PARSEC_M/1e6
        self.assertAlmostEqual(math.degrees(2*math.atan(radius/d["placement_mpc"]))*60,b["angular_major_arcmin"],places=10)
        self.assertGreater(float(v["lookback_myr"]),7000)

    def test_every_regular_template_renders_and_rebuilds(self):
        entries=[]
        for i,family in enumerate(galaxy.FAMILIES):
            b=copy.deepcopy(self.catalog["galaxies"][0])
            b.update(id=f"sample_{i}",body_name=f"Sample{i}",name=f"Sample {i}",template=family)
            entries.append(b)
        self.catalog["galaxies"]=entries
        self.catalog.pop("rendering_overrides",None)
        self.assertEqual(len(galaxy.build_configs(self.catalog,self.root)),len(galaxy.FAMILIES))

    def test_rendering_refinements_survive_source_refresh(self):
        catalog=self.root/"data/galaxies.json"
        catalog.parent.mkdir()
        catalog.write_text(json.dumps(self.catalog))
        shutil.copytree(ROOT/"data/galaxy_sources",self.root/"data/galaxy_sources")
        original=copy.deepcopy(self.catalog["rendering_overrides"])
        with mock.patch.object(fetch,"ROOT",self.root),mock.patch.object(fetch.urllib.request,"urlopen",side_effect=AssertionError("network used")):
            self.assertEqual(fetch.main(["NGC 300","--offline","--refresh","--catalog",str(catalog)]),0)
        self.assertEqual(load_catalog(catalog)["rendering_overrides"],original)

    def test_rendering_refinements_cannot_change_orbit_or_inject_config(self):
        for parameters in [{"referenceBody":"SagittariusA"},{"objectScale":1},{"volumeIntensity":"1\nBody"},
                           {"dustOpacity":float("inf")},{"armCount":2.5},{"enableDustClouds":"false"}]:
            with self.subTest(parameters=parameters),self.assertRaises(CatalogError):
                galaxy.apply_rendering_overrides("volumeIntensity = 1",parameters)

    def test_renderer_exposure_distinguishes_disk_and_old_populations(self):
        b=copy.deepcopy(self.catalog["galaxies"][0])
        b["b_magnitude"]=None
        b["redshift"]=0
        values={family:galaxy.replacements(dict(b,template=family),self.catalog["settings"]) for family in galaxy.FAMILIES}
        # Same observed signal must not be interpreted as equal volumetric emissivity.
        self.assertGreater(float(values["spiral"]["volume_intensity"]),4*float(values["lenticular"]["volume_intensity"]))
        old_color=list(map(float,values["lenticular"]["interarm_color"].split(',')))
        self.assertGreater(old_color[0],old_color[1])
        self.assertGreater(old_color[1],old_color[2])

    def test_saved_ngc300_inclination_reaches_the_rendered_orientation(self):
        body=next(b for b in self.catalog["galaxies"] if b["id"]=="ngc_300")
        configs=galaxy.build_configs(self.catalog,self.root)
        text=configs[self.root/"RealGalacticSystems/Systems/ngc_300.cfg"]
        parameters={}
        for line in text.splitlines():
            if '=' in line:
                key,value=map(str.strip,line.split('=',1))
                if key in {'rotationX','rotationY','rotationZ'}:parameters[key]=float(value)
        rotation=unity_matrix([parameters[k] for k in ['rotationX','rotationY','rotationZ']])
        normal=tuple(rotation[i][1] for i in range(3))
        radial=galaxy.icrs_to_unity(galaxy.sky_basis(body['ra_deg'],body['dec_deg'])[0])
        self.assertAlmostEqual(abs(dot(normal,radial)),math.cos(math.radians(42)),places=10)

    def test_base_and_family_edits_propagate(self):
        before=galaxy.build_configs(self.catalog,self.root)
        path=self.root/self.catalog["templates"]["base"]
        path.write_text(path.read_text().replace("radius = 1","radius = 1\n      contractWeight = 0"))
        after=galaxy.build_configs(self.catalog,self.root)
        self.assertTrue(all("contractWeight = 0" in t for t in after.values()))
        self.assertTrue(all(before[p]!=after[p] for p in before))
        path=self.root/self.catalog["templates"]["types"]["spiral"]
        path.write_text(path.read_text().replace("diskThickness = 6","diskThickness = 7"))
        changed=galaxy.build_configs(self.catalog,self.root)
        self.assertEqual(sum(after[p]!=changed[p] for p in changed),2)

    def test_subset_rebuild_with_other_managed_configs_present(self):
        write_configs(galaxy.build_configs(self.catalog,self.root))
        selected=galaxy.build_configs(self.catalog,self.root,["ngc_300"])
        self.assertEqual(len(selected),1)
        self.assertEqual(next(iter(selected)).stem,"ngc_300")

    def test_unmanaged_output_and_duplicate_existing_body_are_protected(self):
        target=self.root/"RealGalacticSystems/Systems/ngc_300.cfg"
        target.parent.mkdir(parents=True)
        target.write_text("user-authored configuration")
        with self.assertRaises(CatalogError):galaxy.build_configs(self.catalog,self.root)
        target.unlink()
        target=self.root/"RealGalacticSystems/Systems/Existing/galaxy.cfg"
        target.parent.mkdir();target.write_text("Body\n{\n cacheFile = example.bin\n Template { name = Jool }\n name = NGC300\n}\n")
        with self.assertRaises(CatalogError):galaxy.build_configs(self.catalog,self.root)

    def test_invalid_data_and_unsupported_morphologies_fail_before_writes(self):
        for key,value in [("redshift",float("nan")),("mass_solar",-1),("axis_ratio",2),("body_name","Sun"),
                          ("template","irregular"),("name","bad\nBody"),("id","../escape"),("id","con")]:
            c=copy.deepcopy(self.catalog);c["galaxies"][0][key]=value
            with self.subTest(key=key),self.assertRaises(CatalogError):galaxy.build_configs(c,self.root)
        self.assertFalse((self.root/"RealGalacticSystems").exists())
        for morph in ["Im","SBm","Ring","SAB(rs)bc","Sbc pec","merger"]:
            with self.subTest(morph=morph),self.assertRaises(CatalogError):galaxy.classify(morph)

    def test_hubble_stage_letters_are_not_confused_with_bars(self):
        for morph,expected in [("Sb","spiral"),("Sab","spiral"),("SBb","barred_spiral"),
                               ("SABb","weakly_barred_spiral"),("S0/a","lenticular"),
                               ("SB0","barred_lenticular"),("cE2","compact_elliptical"),
                               ("dSph","dwarf_spheroidal"),("cD","elliptical")]:
            self.assertEqual(galaxy.classify(morph),expected,morph)
        for stage in range(8):self.assertEqual(galaxy.classify(f"E{stage}"),"elliptical")

    def test_http_mirror_fallback(self):
        response=io.BytesIO(b'{"data":[]}')
        with mock.patch.object(fetch.urllib.request,"urlopen",side_effect=[urllib.error.URLError("down"),response]) as opener:
            result=fetch.request(["https://first.invalid/query","https://second.invalid/query"],{"name":"NGC 300"})
        self.assertEqual(opener.call_count,2)
        self.assertTrue(result["url"].startswith("https://second.invalid/"))

    def test_saved_raw_records_reproduce_adopted_values_without_network(self):
        for b in self.catalog["galaxies"]:
            raw=load_catalog(ROOT/b["raw_record"])
            with mock.patch.object(fetch.urllib.request,"urlopen",side_effect=AssertionError("network used")):
                adopted=fetch.normalize(raw,self.catalog["settings"])
            self.assertEqual(adopted,b)

    def test_offline_dry_run_writes_nothing(self):
        shutil.copytree(ROOT/"data/galaxy_sources",self.root/"data/galaxy_sources")
        catalog=self.root/"data/galaxies.json"
        catalog.write_text(json.dumps(self.catalog))
        before={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with mock.patch.object(fetch,"ROOT",self.root),mock.patch.object(fetch.urllib.request,"urlopen",side_effect=AssertionError("network used")):
            self.assertEqual(fetch.main(["NGC 300","--offline","--refresh","--dry-run","--catalog",str(catalog)]),0)
        after={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before,after)


class InstalledDiskOrientationTests(unittest.TestCase):
    """Check the shipped disk planes against independent sky projections."""

    def test_every_disk_has_reviewed_orientation_and_frozen_world_axes(self):
        catalog=load_catalog(ROOT/"data/galaxies.json")
        legacy=load_catalog(ROOT/"data/legacy_disk_orientations.json")["galaxies"]
        reviewed={b["body_name"] for b in catalog["galaxies"]+legacy}
        installed=[]
        for path in (ROOT/"RealGalacticSystems").rglob("*.cfg"):
            text=strip_comments(path.read_text(encoding="utf-8-sig"))
            for match in re.finditer(r"(?m)^\s*GalacticDiskDefinition\s*\n",text):
                node=node_contents(text[match.start():],"GalacticDiskDefinition")
                values=dict(re.findall(r"(?m)^\s*(\w+)\s*=\s*([^\r\n]+)",node))
                installed.append(values["planet"].strip())
                self.assertEqual(values["alignRotationToOrbit"].strip(),"false")
                self.assertEqual(float(values["rotationPeriod"]),0)
                self.assertEqual(values["spinAroundDiskAxis"].strip(),"true")
        self.assertEqual(set(installed),reviewed)
        self.assertEqual(len(installed),len(reviewed))

    def test_measured_planes_project_to_adopted_inclination_and_position_angle(self):
        catalog=load_catalog(ROOT/"data/galaxies.json")
        legacy=load_catalog(ROOT/"data/legacy_disk_orientations.json")["galaxies"]
        for body in catalog["galaxies"]+legacy:
            if body.get("orientation_kind")=="galactic_plane":continue
            identifier=body.get("id")
            refinement=catalog["rendering_overrides"].get(identifier,{})
            pa=refinement.get("position_angle_deg",body["position_angle_deg"])
            if "inclination_deg" in body:inc=body["inclination_deg"]
            elif body["template"] in galaxy.SPHEROIDS:inc=0
            else:
                q0=.25 if "lenticular" in body["template"] else .15
                inc=refinement.get("inclination_deg",math.degrees(math.acos(math.sqrt(
                    max(0,(body["axis_ratio"]**2-q0*q0)/(1-q0*q0))))))
            path=ROOT/(body.get("path") or f"RealGalacticSystems/Systems/{identifier}.cfg")
            node=node_contents(strip_comments(path.read_text(encoding="utf-8-sig")),"GalacticDiskDefinition")
            values=dict(re.findall(r"(?m)^\s*(rotation[X-Z])\s*=\s*([^\r\n]+)",node))
            matrix=unity_matrix([float(values[k]) for k in ["rotationX","rotationY","rotationZ"]])
            normal=tuple(matrix[i][1] for i in range(3));major=tuple(matrix[i][0] for i in range(3))
            radial,east,north=map(galaxy.icrs_to_unity,galaxy.sky_basis(body["ra_deg"],body["dec_deg"]))
            with self.subTest(body=body["body_name"]):
                self.assertAlmostEqual(abs(dot(normal,radial)),math.cos(math.radians(inc)),places=10)
                observed_pa=math.degrees(math.atan2(dot(major,east),dot(major,north)))
                self.assertAlmostEqual(math.sin(math.radians(observed_pa-pa)),0,places=10)

    def test_legacy_external_galaxies_are_placed_on_the_observed_sightline(self):
        for body in load_catalog(ROOT/"data/legacy_disk_orientations.json")["galaxies"]:
            if body["orientation_kind"]!="sky_disk":continue
            text=strip_comments((ROOT/body["path"]).read_text(encoding="utf-8-sig"))
            node=node_contents(text,"Orbit")
            values=dict(re.findall(r"(?m)^\s*(\w+)\s*=\s*([^\r\n]+)",node))
            orbit={k:v.strip() if k in {"referenceBody","mode"} else float(v) for k,v in values.items()}
            self.assertEqual(orbit["referenceBody"],"Sun")
            self.assertEqual(orbit["period"],3e300)
            self.assertEqual(orbit["mode"],"OFF")
            self.assertAlmostEqual(orbit["semiMajorAxis"]/body["distance_m"],1,places=14)
            properties=node_contents(text,"Properties")
            soi=float(re.search(r"(?m)^\s*sphereOfInfluence\s*=\s*(\S+)",properties)[1])
            self.assertLess(soi/body["distance_m"],0.001)
            if "core_soi_pc" in body:
                self.assertAlmostEqual(soi/(body["core_soi_pc"]*PARSEC_M),1,places=14)
            p=orbit_position(orbit);direction=tuple(v/math.hypot(*p) for v in p)
            expected=galaxy.icrs_to_unity(galaxy.sky_basis(body["ra_deg"],body["dec_deg"])[0])
            self.assertLess(math.dist((direction[0],direction[2],direction[1]),expected),1e-14)
            self.assertEqual(orbit_position(orbit,1e15),p)

    def test_milky_way_plane_and_near_bar_end(self):
        record=next(b for b in load_catalog(ROOT/"data/legacy_disk_orientations.json")["galaxies"]
                    if b["body_name"]=="SagittariusA")
        node=node_contents(strip_comments((ROOT/record["path"]).read_text()),"GalacticDiskDefinition")
        v=dict(re.findall(r"(?m)^\s*(\w+)\s*=\s*([^\r\n]+)",node))
        matrix=unity_matrix([float(v[k]) for k in ["rotationX","rotationY","rotationZ"]])
        self.assertLess(math.dist(tuple(matrix[i][1] for i in range(3)),(0,1,0)),1e-14)
        # GalacticDisk's local bar axis is (cos(a),0,sin(a)), opposite Unity yaw.
        a=math.radians(float(v["barAngle"]))
        bar=tuple(matrix[i][0]*math.cos(a)+matrix[i][2]*math.sin(a) for i in range(3))
        near=bar if bar[0]<0 else tuple(-x for x in bar)
        self.assertGreater(near[2],0)  # Galactic longitude > 0 at the near bar end.
        self.assertAlmostEqual(math.degrees(math.acos(-near[0])),30,places=10)


if __name__=="__main__":unittest.main()
