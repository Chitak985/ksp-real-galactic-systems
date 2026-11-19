"""Bulk source integrity, scientific gates, identity checks, and offline previews."""
import copy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import fetch_bulk_galaxies as bulk
import rebuild_galaxies as builder
from rebuild_configs import CatalogError, load_catalog

ROOT = Path(__file__).resolve().parents[1]


class BulkGalaxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT/"data/galaxies.json")
        cls.snapshot = load_catalog(ROOT/bulk.SNAPSHOT)
        cls.report = load_catalog(ROOT/bulk.REPORT)
        cls.ids = set(cls.report["selected_ids"])
        cls.bodies = [b for b in cls.catalog["galaxies"] if b["id"] in cls.ids]
        cls.cosmic = [b for b in cls.bodies if "lvdb_key" not in b]
        cls.row = load_catalog(ROOT/cls.cosmic[0]["raw_record"])["simbad"]["rows"][0]

    def test_snapshot_integrity_and_derived_rows_are_verified(self):
        self.assertEqual(len(bulk.validate_snapshot(self.snapshot)), 4000)
        for key in ("text", "query"):
            changed = copy.deepcopy(self.snapshot)
            changed[key] += " "
            with self.assertRaises(CatalogError):
                bulk.validate_snapshot(changed)
        changed = copy.deepcopy(self.snapshot)
        changed["rows"][0]["ra"] += 1
        with self.assertRaises(CatalogError):
            bulk.validate_snapshot(changed)

    def test_quality_ambiguity_photometry_and_redshift_gates(self):
        for key, value in [("rvz_nature", "p"), ("rvz_qual", "D"), ("galdim_qual", "E"),
                           ("morph_type", "S?"), ("morph_type", "Irr"), ("B", None), ("K", 99),
                           ("rvz_redshift", .001), ("galdim_bibcode", None), ("galdim_minaxis", 0)]:
            row = dict(self.row, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(CatalogError):
                bulk.normalize_row(row, self.snapshot, self.catalog["settings"])

    def test_current_batch_has_saved_sources_and_small_sois(self):
        expected=self.report["bulk_count"]+self.report["nearby_count"]
        self.assertEqual(expected,self.report.get("requested_count",expected))
        self.assertEqual(len(self.ids), expected)
        self.assertEqual(len(self.bodies), expected)
        self.assertEqual(len(self.cosmic), self.report["bulk_count"])
        aliases = set()
        for body in self.catalog["galaxies"]:
            identity = bulk.nearby.canonical(body["body_name"])
            self.assertNotIn(identity, aliases)
            aliases.add(identity)
        for body in self.bodies:
            with self.subTest(galaxy=body["id"]):
                raw = load_catalog(ROOT/body["raw_record"])
                if "lvdb_key" not in body:
                    adopted, _ = bulk.normalize_row(raw["simbad"]["rows"][0], self.snapshot, self.catalog["settings"])
                    self.assertEqual(adopted, body)
                    self.assertEqual(raw["bulk_snapshot_sha256"], self.snapshot["sha256"])
                text = (ROOT/"RealGalacticSystems/Systems"/(body["id"]+".cfg")).read_text()
                self.assertIn("starCount = 2000", text)
                self.assertIn("referenceBody = Sun", text)
                self.assertIn("period = 3E300", text)
                self.assertIn("mode = OFF", text)
                values = builder.replacements(body, self.catalog["settings"])
                self.assertLessEqual(float(values["soi"])/builder.PARSEC_M, 100.00000001)

    def test_sky_identity_guard_handles_ra_wrap_and_nearby_distinct_objects(self):
        a = {"ra_deg": 359.9999, "dec_deg": 20}
        self.assertTrue(bulk.same_sightline(a, {"ra_deg": .0001, "dec_deg": 20}))
        self.assertFalse(bulk.same_sightline(a, {"ra_deg": .1, "dec_deg": 20}))

    def test_star_sampling_budget_is_bounded(self):
        for value in (-1, 2000001, 2.5, float("nan")):
            with self.assertRaises(CatalogError):
                builder.apply_rendering_overrides("starCount = 30000", {"starCount": value})
        self.assertIn("starCount = 2000", builder.apply_rendering_overrides("starCount = 30000", {"starCount": 2000}))

    def test_previous_batches_keep_their_original_source_snapshots(self):
        batches=[*self.report.get("previous_selections",[]),self.report]
        self.assertGreaterEqual(len(batches),2)
        bodies={b["id"]:b for b in self.catalog["galaxies"]}
        sources={}
        def source(path):
            if path not in sources:
                sources[path]=load_catalog(ROOT/path)
                bulk.validate_snapshot(sources[path])
            return sources[path]
        for report in batches:
            snapshot=source(report["snapshot"])
            self.assertEqual(snapshot["sha256"],report["snapshot_sha256"])
            for identity in report["selected_ids"]:
                body=bodies[identity]
                raw=load_catalog(ROOT/body["raw_record"])
                if "bulk_snapshot" in raw:
                    original=source(raw["bulk_snapshot"])
                    self.assertEqual(original["sha256"],raw["bulk_snapshot_sha256"])
                    self.assertEqual(original["sha256"],report["snapshot_sha256"])

    def test_cosmology_cache_preserves_input_isolation_and_result_ownership(self):
        settings=self.catalog["settings"]["cosmology"]
        original=builder.cosmology(.02,settings)
        altered=dict(settings,h0_km_s_mpc=settings["h0_km_s_mpc"]*2)
        self.assertAlmostEqual(builder.cosmology(.02,altered)["placement_mpc"],original["placement_mpc"]/2)
        result=builder.cosmology(.02,settings)
        result["placement_mpc"]=-1
        self.assertEqual(builder.cosmology(.02,settings),original)
        with self.assertRaises(CatalogError):
            builder.cosmology(.02,dict(settings,omega_m=float("nan")))

    def test_bulk_offline_dry_run_leaves_all_files_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT/"templates/galaxies", root/"templates/galaxies")
            for relative in (bulk.SNAPSHOT, "data/galaxies.json"):
                dest = root/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/relative, dest)
            catalog = copy.deepcopy(self.catalog)
            omitted = self.cosmic[0]["id"]
            catalog["galaxies"] = [b for b in catalog["galaxies"] if b["id"] != omitted]
            catalog["rendering_overrides"].pop(omitted)
            import json
            (root/"data/galaxies.json").write_text(json.dumps(catalog))
            before = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            with mock.patch.object(bulk, "ROOT", root), mock.patch.object(bulk, "select_nearby", return_value=([], [])), \
                 mock.patch.object(bulk.fetch.urllib.request, "urlopen", side_effect=AssertionError("network used")):
                self.assertEqual(bulk.main(["--count", "1", "--offline", "--dry-run"]), 0)
            after = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
