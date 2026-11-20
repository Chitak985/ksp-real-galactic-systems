"""LVDB units, confirmation filters, measured shape conversion, and safe imports."""

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import fetch_galaxies
import fetch_nearby_galaxies as nearby
import rebuild_galaxies as galaxy
from rebuild_configs import CatalogError, PARSEC_M, load_catalog

ROOT=Path(__file__).resolve().parents[1]


class NearbyGalaxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=load_catalog(ROOT/"data/galaxies.json")
        cls.review=load_catalog(ROOT/"data/nearby_galaxy_review.json")
        cls.snapshot=load_catalog(ROOT/"data/galaxy_sources/lvdb_nearby_snapshot.json")
        cls.report=load_catalog(ROOT/"data/nearby_galaxy_selection.json")
        cls.ids={body["id"] for body in cls.catalog["galaxies"] if "lvdb_key" in body}
        cls.raw=load_catalog(ROOT/"data/galaxy_sources/lvdb_segue_1.json")

    def test_released_source_checksum_and_version_are_required(self):
        rows=nearby.snapshot_rows(self.snapshot,self.review)
        self.assertEqual(len(rows),1733)
        for key,value in [("text",self.snapshot["text"]+"\n"),("version","other")]:
            source=copy.copy(self.snapshot);source[key]=value
            with self.assertRaises(CatalogError):nearby.snapshot_rows(source,self.review)

    def test_heliocentric_distance_mass_logarithm_and_measured_v_band(self):
        body=nearby.normalize_lvdb(self.raw,self.catalog["settings"])
        row=self.raw["row"]
        self.assertEqual(body["distance"]["mpc"]*1000,float(row["distance"]))
        self.assertNotEqual(body["distance"]["mpc"]*1000,float(row["distance_gc"]))
        self.assertAlmostEqual(body["mass_solar"],10**float(row["mass_stellar"]))
        self.assertEqual(body["v_magnitude"],float(row["apparent_magnitude_v"]))
        self.assertIsNone(body["b_magnitude"])
        self.assertEqual(body,fetch_galaxies.normalize(self.raw,self.catalog["settings"]))

    def test_gaussian_projection_recovers_all_saved_measured_half_light_sizes(self):
        bodies=[b for b in self.catalog["galaxies"] if b["id"] in self.ids]
        self.assertGreaterEqual(len(bodies),50)
        for body in bodies:
            values=galaxy.replacements(body,self.catalog["settings"])
            sigma_m=float(values["spheroid_sigma"])*float(values["object_scale"])*6000
            d=body["distance"]["mpc"]*1e6*PARSEC_M
            half_arcmin=math.degrees(math.atan(sigma_m*math.sqrt(2*math.log(2))/d))*60
            with self.subTest(body=body["id"]):
                self.assertAlmostEqual(half_arcmin,body["half_light_major_arcmin"],places=10)
                self.assertAlmostEqual(float(values["spheroid_minor_sigma"])/float(values["spheroid_sigma"]),body["axis_ratio"])
                self.assertLess(body["angular_major_arcmin"],600)

    def test_confirmation_missing_data_and_invalid_shapes_fail(self):
        for key,value in [("confirmed_galaxy","0"),("confirmed_real","0"),("confirmed_star_cluster","1"),
                          ("distance","-1"),("rhalf",""),("mass_stellar","nan"),("ellipticity","1"),
                          ("position_angle",""),("vlos_systemic","")]:
            raw=copy.deepcopy(self.raw);raw["row"][key]=value
            with self.subTest(key=key),self.assertRaises(CatalogError):
                nearby.normalize_lvdb(raw,self.catalog["settings"])

    def test_circular_proxy_requires_a_tight_observed_ellipticity_limit(self):
        raw=load_catalog(ROOT/"data/galaxy_sources/lvdb_crater_2.json")
        body=nearby.normalize_lvdb(raw,self.catalog["settings"])
        self.assertEqual(body["axis_ratio"],1)
        self.assertIn("upper limit",body["uncertainty_note"])
        raw["row"]["ellipticity_ul"]="0.5"
        with self.assertRaises(CatalogError):nearby.normalize_lvdb(raw,self.catalog["settings"])

    def test_fifty_additions_are_the_nearest_eligible_new_galaxies(self):
        catalog=copy.deepcopy(self.catalog)
        catalog["galaxies"]=[b for b in catalog["galaxies"] if b["id"] not in self.ids]
        rows=nearby.snapshot_rows(self.snapshot,self.review)
        selected,skipped=nearby.select_nearest(rows,self.review,self.snapshot,catalog,50)
        saved=sorted((b for b in self.catalog["galaxies"] if b["id"] in self.ids and "redshift_status" not in b),
                     key=lambda b:(b["distance"]["mpc"],b["id"]))[:50]
        self.assertEqual([b["id"] for b,raw in selected],[b["id"] for b in saved])
        distances=[b["distance"]["mpc"] for b,raw in selected]
        self.assertEqual(distances,sorted(distances))
        self.assertEqual(sum(d*1e6*nearby.LY_PER_KPC/1000<=100000 for d in distances),3)
        self.assertTrue({"draco_2","carina_3","willman_1","sagittarius_1"}.isdisjoint(self.ids))
        # A padded alternate designation must not add the existing NGC 185 again.
        row=next(row for row in rows if row["key"]=="ngc_0185")
        with self.assertRaises(CatalogError):nearby.select_nearest([row],self.review,self.snapshot,catalog,1)

    def test_v_photometry_and_profile_extent_validation(self):
        body=nearby.normalize_lvdb(self.raw,self.catalog["settings"])
        actual=galaxy.replacements(body,self.catalog["settings"])
        no_flux=dict(body,v_magnitude=None)
        self.assertLess(float(actual["volume_intensity"]),float(galaxy.replacements(no_flux,self.catalog["settings"])["volume_intensity"]))
        for value in [0,7,float("nan")]:
            with self.assertRaises(CatalogError):galaxy.replacements(dict(body,spheroid_extent_sigma=value),self.catalog["settings"])
        spiral=next(b for b in self.catalog["galaxies"] if b["template"]=="spiral")
        with self.assertRaises(CatalogError):galaxy.replacements(dict(spiral,spheroid_extent_sigma=3),self.catalog["settings"])

    def test_offline_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/"templates/galaxies",root/"templates/galaxies")
            for rel in ["data/galaxies.json","data/nearby_galaxy_review.json","data/galaxy_sources/lvdb_nearby_snapshot.json"]:
                target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
            # Keep one known measured object available even after all LVDB
            # candidates have been installed by a subsequent bulk batch.
            catalog=load_catalog(root/"data/galaxies.json")
            catalog["galaxies"]=[b for b in catalog["galaxies"] if b["id"]!="segue_1"]
            catalog["rendering_overrides"].pop("segue_1",None)
            (root/"data/galaxies.json").write_text(json.dumps(catalog))
            before={p.relative_to(root):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            with mock.patch.object(nearby,"ROOT",root),mock.patch.object(nearby.urllib.request,"urlopen",side_effect=AssertionError("network used")):
                self.assertEqual(nearby.main(["--count","1","--offline","--dry-run"]),0)
            after={p.relative_to(root):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before,after)

    def test_followup_import_keeps_existing_objects_and_selection_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/"templates/galaxies",root/"templates/galaxies")
            original_config="RealGalacticSystems/Systems/segue_1.cfg"
            for rel in ["data/galaxies.json","data/nearby_galaxy_review.json",
                        "data/galaxy_sources/lvdb_nearby_snapshot.json",
                        "data/nearby_galaxy_selection.json",original_config]:
                target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
            catalog=load_catalog(root/"data/galaxies.json")
            available=next(b for b in catalog["galaxies"] if "lvdb_key" in b and b["id"]!="segue_1")
            catalog["galaxies"]=[b for b in catalog["galaxies"] if b["id"]!=available["id"]]
            catalog["rendering_overrides"].pop(available["id"],None)
            (root/"data/galaxies.json").write_text(json.dumps(catalog))
            original_bytes=(root/original_config).read_bytes()
            with mock.patch.object(nearby,"ROOT",root),mock.patch.object(nearby.urllib.request,"urlopen",side_effect=AssertionError("network used")):
                self.assertEqual(nearby.main(["--count","1","--offline"]),0)
            updated=load_catalog(root/"data/galaxies.json")
            saved={b["id"]:b for b in updated["galaxies"]}
            self.assertEqual(len(saved),len(catalog["galaxies"])+1)
            for body in catalog["galaxies"]:self.assertEqual(saved[body["id"]],body)
            self.assertEqual((root/original_config).read_bytes(),original_bytes)
            report=load_catalog(root/"data/nearby_galaxy_selection.json")
            self.assertEqual(len(report["selected_ids"]),1)
            self.assertEqual(report["previous_selections"][-1]["selected_ids"],self.report["selected_ids"])
            added=saved[report["selected_ids"][0]]
            self.assertTrue((root/added["raw_record"]).is_file())
            outputs=galaxy.build_configs(updated,root,[added["id"]])
            for path,text in outputs.items():self.assertEqual(path.read_text(encoding="utf-8"),text)


if __name__=="__main__":unittest.main()
