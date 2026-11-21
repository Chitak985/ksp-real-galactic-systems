"""Reviewed missing data, explicit velocity proxies, and exact requested selection."""
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import add_reviewed_galaxies as reviewed
import fetch_nearby_galaxies as nearby
import rebuild_galaxies as builder
from rebuild_configs import CatalogError,load_catalog

ROOT=Path(__file__).resolve().parents[1]


class ReviewedGalaxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=load_catalog(ROOT/"data/galaxies.json")
        cls.by_id={b["id"]:b for b in cls.catalog["galaxies"]}
        cls.report=load_catalog(ROOT/reviewed.REPORT)
        cls.ids=set(cls.report["selected_ids"])

    def test_exact_requested_batch_and_distant_boundary(self):
        self.assertEqual(len(self.ids),57)
        self.assertEqual(len(self.report["nearby_ids"]),15)
        self.assertEqual(len(self.report["far_ids"]),42)
        for key in self.report["far_ids"]:
            body=self.by_id[key]
            self.assertGreater(builder.distance_measures(body,self.catalog["settings"])["placement_mpc"],150)
            self.assertNotIn("redshift_status",body)

    def test_original_rows_survive_and_reviewed_records_replay(self):
        for key in self.report["nearby_ids"]:
            raw=load_catalog(ROOT/self.by_id[key]["raw_record"])
            before=copy.deepcopy(raw)
            self.assertFalse(raw["row"]["vlos_systemic"])
            self.assertTrue(raw["literature_review"]["sources"])
            self.assertEqual(nearby.normalize_lvdb(raw,self.catalog["settings"]),{k:v for k,v in self.by_id[key].items() if k!="far"})
            self.assertEqual(raw,before)

    def test_unknown_redshifts_are_null_with_approximate_colors_only(self):
        unknown=[self.by_id[key] for key in self.report["nearby_ids"] if self.by_id[key]["redshift_status"]=="unmeasured"]
        self.assertEqual(len(unknown),13)
        for body in unknown:
            self.assertIsNone(body["redshift"])
            values=builder.replacements(body,self.catalog["settings"])
            self.assertIn("radial velocity unmeasured",values["description"])
            self.assertNotIn("z=0",values["description"])
            self.assertIn("unshifted color approximation",values["description"])
            self.assertGreater(builder.distance_measures(body,self.catalog["settings"])["lookback_years"],0)
        body=copy.deepcopy(unknown[0])
        body["distance"]={"kind":"cosmological_redshift","cosmological_redshift":.02}
        with self.assertRaises(CatalogError):builder.replacements(body,self.catalog["settings"])
        body=copy.deepcopy(unknown[0]);body["redshift"]=0
        with self.assertRaises(CatalogError):builder.replacements(body,self.catalog["settings"])

    def test_cluster_and_flow_proxies_are_labeled_and_do_not_set_distance(self):
        for key,expected,status in [("ikn",38,"cluster_proxy"),("centaurus_c",300,"model_proxy")]:
            body=self.by_id[key]
            self.assertEqual(body["redshift_status"],status)
            self.assertAlmostEqual(body["redshift"]*builder.C_KM_S,expected)
            values=builder.replacements(body,self.catalog["settings"])
            self.assertIn(status.replace("_"," "),values["description"])
            self.assertNotEqual(builder.distance_measures(body,self.catalog["settings"])["placement_mpc"],expected/67.4)

    def test_published_uncertainties_host_distance_and_resolved_geometry(self):
        self.assertEqual(self.by_id["leo_k"]["distance"]["uncertainty_minus_mpc"],.127)
        self.assertEqual(self.by_id["leo_m"]["distance"]["uncertainty_plus_mpc"],.021)
        self.assertEqual(self.by_id["andromeda_36"]["distance"]["kind"],"adopted_local")
        self.assertIsNone(self.by_id["andromeda_36"]["distance"]["uncertainty_minus_mpc"])
        self.assertAlmostEqual(self.by_id["donatiello_8"]["half_light_major_arcmin"],19/60)
        self.assertAlmostEqual(self.by_id["donatiello_8"]["axis_ratio"],.36)
        self.assertAlmostEqual(self.by_id["centaurus_c"]["mass_solar"],10**5.23)
        self.assertIn("star-formation-history",self.by_id["centaurus_c"]["mass_kind"])

    def test_reviews_cannot_bypass_confirmation_or_discard_catalog_velocity(self):
        raw=load_catalog(ROOT/self.by_id["leo_k"]["raw_record"])
        raw["row"]["confirmed_galaxy"]="0"
        with self.assertRaises(CatalogError):nearby.normalize_lvdb(raw,self.catalog["settings"])
        raw=load_catalog(ROOT/self.by_id["leo_k"]["raw_record"])
        raw["row"]["vlos_systemic"]="100"
        with self.assertRaises(CatalogError):nearby.normalize_lvdb(raw,self.catalog["settings"])
        raw=load_catalog(ROOT/self.by_id["leo_k"]["raw_record"])
        raw["literature_review"]["row_updates"]["confirmed_galaxy"]="1"
        with self.assertRaises(CatalogError):nearby.normalize_lvdb(raw,self.catalog["settings"])

    def test_offline_selection_replays_all_fifty_seven_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/"templates/galaxies",root/"templates/galaxies")
            for relative in [reviewed.REVIEW,"data/nearby_galaxy_review.json",
                             "data/galaxy_sources/lvdb_nearby_snapshot.json",reviewed.bulk.SNAPSHOT]:
                dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
            catalog=copy.deepcopy(self.catalog)
            catalog["galaxies"]=[b for b in catalog["galaxies"] if b["id"] not in self.ids]
            catalog["rendering_overrides"]={k:v for k,v in catalog["rendering_overrides"].items() if k not in self.ids}
            (root/"data/galaxies.json").write_text(json.dumps(catalog))
            near,far,snapshot,skipped=reviewed.candidates(catalog,root)
            self.assertEqual({body["id"] for body,raw in near+far},self.ids)
            before={p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with mock.patch.object(reviewed,"ROOT",root),mock.patch.object(reviewed.bulk.fetch.urllib.request,"urlopen",side_effect=AssertionError("network used")):
                self.assertEqual(reviewed.main(["--dry-run"]),0)
            after={p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before,after)


if __name__=="__main__":unittest.main()
