import copy
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import rebuild_configs as generator


ROOT = Path(__file__).resolve().parents[1]


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = generator.load_catalog(ROOT / "data/bodies.json")
        # Keep the compiler's small editable fixture independent of catalog growth.
        self.catalog["bodies"] = [body for body in self.catalog["bodies"]
                                  if body["id"] == "tau_ceti"]
        for relative in self.catalog["templates"].values():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)

    def configs(self):
        return generator.build_configs(self.catalog, self.root)

    def test_tau_ceti_physics_science_and_layout(self):
        configs = self.configs()
        self.assertEqual(len(configs), 1)
        path, text = next(iter(configs.items()))
        self.assertEqual(path.relative_to(self.root).as_posix(),
                         "RealGalacticSystems/Systems/MilkyWay/Systems/tau_ceti.cfg")
        self.assertIn("name = Tau_Ceti", text)
        self.assertIn("mass = 1.3720443E30", text)
        self.assertIn("radius = 551690100", text)
        self.assertIn("rotationPeriod = 3974400", text)
        self.assertIn("luminosity = 612", text)
        self.assertIn("flyingAltitudeThreshold = 55169.01", text)
        self.assertIn("spaceAltitudeThreshold = 1655070300", text)
        self.assertIn("ambientTemp = 5320", text)
        self.assertIn("referenceBody = Sun", text)
        self.assertIn("period = 3E300", text)
        self.assertIn("mode = OFF", text)
        self.assertNotIn("sunlightShadowStrength", text)
        self.assertNotIn("{{", text)
        for line in text.splitlines():
            if "recoveryValue =" in line:
                recovery = int(line.split("=", 1)[1].split("//", 1)[0])
                self.assertEqual(recovery, 62)
        self.assertEqual(self.catalog["bodies"][0]["position"]["epoch"], "J2000")

    def test_position_round_trip_preserves_sol_distance(self):
        body = self.catalog["bodies"][0]
        settings = self.catalog["settings"]
        heliocentric = generator.heliocentric_position(body["position"])
        orbit = generator.frozen_orbit(heliocentric, settings)
        self.assertEqual(orbit["referenceBody"], "Sun")
        expected_distance = body["position"]["distance_pc"] * generator.PARSEC_M
        self.assertAlmostEqual(orbit["semiMajorAxis"] / expected_distance, 1, places=14)
        actual = generator.orbit_position(orbit)
        self.assertLess(math.dist(heliocentric, actual), 1000)
        distance_pc = math.hypot(*actual) / generator.PARSEC_M
        self.assertAlmostEqual(distance_pc, body["position"]["distance_pc"], places=9)
        future = generator.orbit_position(orbit, 1e12)
        self.assertEqual(future, actual)
        longitude = math.degrees(math.atan2(heliocentric[1], heliocentric[0])) % 360
        latitude = math.degrees(math.atan2(heliocentric[2], math.hypot(*heliocentric[:2])))
        # Independent catalog coordinates from SIMBAD.
        self.assertAlmostEqual(longitude, 173.1006757445964, places=4)
        self.assertAlmostEqual(latitude, -73.4397246835404, places=4)

    def test_recovery_value_rounds_up_without_incrementing_whole_values(self):
        body = self.catalog["bodies"][0]
        for distance_ly, expected in [(1, "51"), (1.01, "52"), (1.99, "52")]:
            with self.subTest(distance_ly=distance_ly):
                body["position"]["distance_pc"] = distance_ly * generator.LIGHT_YEAR_M / generator.PARSEC_M
                values = generator.replacements(body, self.catalog["settings"])
                self.assertEqual(values["recovery_value"], expected)

    def test_circular_placement_handles_axes_poles_and_quadrants(self):
        for point in [(1e17, 0, 0), (0, -1e17, 0), (0, 0, 1e17),
                      (0, 0, -1e17), (-1e17, 2e17, -3e17)]:
            with self.subTest(point=point):
                orbit = generator.frozen_orbit(point, self.catalog["settings"])
                self.assertLess(math.dist(point, generator.orbit_position(orbit)), 1000)

    def test_template_edits_propagate_to_every_catalogued_star(self):
        second = copy.deepcopy(self.catalog["bodies"][0])
        second.update(id="demo_star", name="Demo-Star", radius_m=700000000)
        self.catalog["bodies"].append(second)
        template = self.root / self.catalog["templates"]["sun_like"]
        template.write_text(template.read_text().replace("contractWeight = 5", "contractWeight = 9"))
        configs = self.configs()
        self.assertEqual({p.name for p in configs}, {"tau_ceti.cfg", "demo_star.cfg"})
        for text in configs.values():
            self.assertIn("contractWeight = 9", text)
        demo = next(text for path, text in configs.items() if path.name == "demo_star.cfg")
        self.assertIn("name = Demo_Star", demo)
        self.assertIn("flyingAltitudeThreshold = 70000", demo)

    def test_rebuild_overwrites_only_managed_files_and_is_repeatable(self):
        relative = "RealGalacticSystems/Systems/MilkyWay/Configs/SagittariusA.cfg"
        parent = self.root / relative
        parent.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, parent)
        original_parent = parent.read_bytes()
        readme = self.root / "README.md"
        readme.write_text("Keep this README unchanged.")
        original_readme = readme.read_bytes()
        unmanaged = self.root / "RealGalacticSystems/Systems/MilkyWay/Systems/keep.cfg"
        unmanaged.parent.mkdir(parents=True, exist_ok=True)
        unmanaged.write_text("Unmanaged content")
        configs = self.configs()
        generator.write_configs(configs)
        first = {p: p.read_bytes() for p in configs}
        generator.write_configs(self.configs())
        self.assertEqual(first, {p: p.read_bytes() for p in configs})
        self.assertEqual(parent.read_bytes(), original_parent)
        self.assertEqual(readme.read_bytes(), original_readme)
        self.assertEqual(unmanaged.read_text(), "Unmanaged content")
        self.assertFalse(list(unmanaged.parent.glob("*.tmp")))

    def test_invalid_catalog_does_not_modify_existing_config(self):
        configs = self.configs()
        generator.write_configs(configs)
        path = next(iter(configs))
        before = path.read_bytes()
        second = copy.deepcopy(self.catalog["bodies"][0])
        second.update(id="invalid", name="Invalid", mass_kg=-1)
        self.catalog["bodies"].append(second)
        with self.assertRaises(generator.CatalogError):
            generator.write_configs(self.configs())
        self.assertEqual(path.read_bytes(), before)

    def test_unknown_placeholders_duplicate_names_and_unsupported_systems_fail(self):
        for change in ["placeholder", "duplicate", "binary", "galaxy", "long_description", "nonfinite"]:
            with self.subTest(change=change):
                catalog = copy.deepcopy(self.catalog)
                if change == "placeholder":
                    template = self.root / catalog["templates"]["sun_like"]
                    original = template.read_text()
                    template.write_text(original + "\n{{misspelled_radius}}\n")
                    self.addCleanup(template.write_text, original)
                elif change == "duplicate":
                    duplicate = copy.deepcopy(catalog["bodies"][0])
                    duplicate.update(id="another_id", name="tau ceti")
                    catalog["bodies"].append(duplicate)
                elif change == "binary":
                    catalog["bodies"][0]["system_type"] = "binary"
                elif change == "galaxy":
                    catalog["bodies"][0]["galaxy"] = "Andromeda"
                elif change == "long_description":
                    catalog["bodies"][0]["description"] = "x" * 100
                else:
                    catalog["bodies"][0]["temperature_k"] = float("nan")
                with self.assertRaises(generator.CatalogError):
                    generator.build_configs(catalog, self.root)
                if change == "placeholder":
                    template.write_text(original)

    def test_selected_body_and_json_errors(self):
        self.assertEqual(len(generator.build_configs(self.catalog, self.root, ["tau_ceti"])), 1)
        with self.assertRaises(generator.CatalogError):
            generator.build_configs(self.catalog, self.root, ["missing"])
        path = self.root / "invalid.json"
        for text in ['{"schema_version":1,"schema_version":2}', '{"mass":NaN}']:
            path.write_text(text)
            with self.assertRaises(generator.CatalogError):
                generator.load_catalog(path)


class SavedCatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = generator.load_catalog(ROOT / "data/bodies.json")

    def test_catalog_has_complete_provenance_and_distance_order(self):
        bodies = self.catalog["bodies"]
        original = {"tau_ceti", "82_g_eridani", "beta_canum_venaticorum", "61_virginis",
                    "beta_comae_berenices", "kappa1_ceti", "61_ursae_majoris",
                    "iota_persei", "lambda_serpentis", "lambda_aurigae"}
        self.assertLessEqual(original, {body["id"] for body in bodies})
        self.assertEqual(bodies[0]["id"], "tau_ceti")
        distances = [body["position"]["distance_pc"] for body in bodies]
        self.assertEqual(distances, sorted(distances))
        required = {"radius_m", "mass_kg", "temperature_k", "luminosity_solar",
                    "rotation_period_days", "position", "main_color"}
        for body in bodies:
            with self.subTest(body=body["id"]):
                self.assertEqual(body["system_type"], "single")
                self.assertEqual(body["galaxy"], "MilkyWay")
                covered = {field for source in body["sources"] for field in source["fields"]}
                self.assertLessEqual(required, covered)
                for field in required - {"position", "main_color"}:
                    self.assertTrue(math.isfinite(body[field]) and body[field] > 0)
                if body["id"] != "tau_ceti":
                    self.assertIn(body["rotation_period_kind"],
                                  {"observed_modulation", "activity_calibration", "gyrochronology_estimate"})
                    source = next(source for source in body["sources"]
                                  if "rotation_period_days" in source["fields"])
                    self.assertTrue(source.get("method"))
        estimated = {body["id"] for body in bodies
                     if body.get("rotation_period_kind") == "activity_calibration"}
        self.assertLessEqual({"beta_canum_venaticorum", "iota_persei", "lambda_aurigae"}, estimated)

    def test_physical_units_match_the_saved_published_values(self):
        for body in self.catalog["bodies"]:
            with self.subTest(body=body["id"]):
                source = next(source for source in body["sources"] if "mass_kg" in source["fields"])
                reported = {**source["reported_values"], **source.get("derived_values", {})}
                constants = source["conversion_constants"]
                self.assertAlmostEqual(body["radius_m"] / constants["solar_radius_m"],
                                       reported["radius_solar"], places=12)
                self.assertAlmostEqual(body["mass_kg"] / constants["solar_mass_kg"],
                                       reported["mass_solar"], places=12)
                luminosity = reported.get("luminosity_solar")
                if luminosity is None:
                    luminosity = 10 ** reported["log10_luminosity_solar"]
                self.assertAlmostEqual(body["luminosity_solar"], luminosity, places=12)
                self.assertEqual(body["temperature_k"], reported["temperature_k"])
                rotation = next(source for source in body["sources"]
                                if "rotation_period_days" in source["fields"])
                self.assertEqual(body["rotation_period_days"],
                                 {**rotation.get("reported_values", {}),
                                  **rotation.get("derived_values", {})}["rotation_period_days"])

    def test_estimated_candidates_preserve_uncertainty_and_model_provenance(self):
        candidates = [body for body in self.catalog["bodies"]
                      if body.get("data_status") == "estimated_candidate"]
        self.assertTrue(candidates)
        for body in candidates:
            with self.subTest(body=body["id"]):
                self.assertEqual(body["multiplicity_status"], "no_known_stellar_companion")
                self.assertLess(len(body["description"]), 100)
                self.assertIn("estimated", body["description"])
                self.assertIn("single status unconfirmed", body["description"])
                self.assertIn(body["rotation_period_kind"], {"gyrochronology_estimate", "observed_modulation"})
                self.assertIsInstance(body["identifiers"]["gaia_dr3"], str)
                physical = body["sources"][0]
                values = physical["reported_values"]
                constants = physical["conversion_constants"]
                # Independently verify the model gravity/radius and bolometric relation.
                gravity_si = 10 ** values["logg_cgs"] / 100
                radius = math.sqrt(constants["gravitational_constant_si"] * body["mass_kg"]
                                   / gravity_si) / constants["solar_radius_m"]
                self.assertEqual(round(radius, 4), physical["derived_values"]["radius_solar"])
                luminosity = (body["radius_m"] / constants["solar_radius_m"]) ** 2
                luminosity *= (body["temperature_k"] / constants["solar_temperature_k"]) ** 4
                self.assertEqual(round(luminosity, 4), body["luminosity_solar"])
                self.assertEqual(body["temperature_k"], round(values["temperature_k"]))
                distance = physical["reported_percentiles"]["distance_kpc"]
                self.assertLessEqual(distance["5"], distance["50"])
                self.assertLessEqual(distance["50"], distance["95"])
                self.assertEqual(body["position"]["distance_pc"], distance["50"] * 1000)
                region = body["galactic_region"]
                if region["kind"] == "outer_disk_candidate":
                    self.assertGreaterEqual(region["cylindrical_radius_kpc"], 12)
                    self.assertGreaterEqual(region["radius_over_distance_90_percent_posterior_kpc"]["lower"], 11.5)
                elif region["kind"] == "disk_gap_candidate":
                    self.assertGreaterEqual(region["cylindrical_radius_kpc"], 8.2)
                    self.assertLess(region["cylindrical_radius_kpc"], 12.2)
                    lower, upper = region["gap_band_distance_kpc"]
                    self.assertGreaterEqual(body["position"]["distance_pc"] / 1000, lower)
                    self.assertLess(body["position"]["distance_pc"] / 1000, upper)
                else:
                    self.fail("Estimated candidate has an undocumented Galactic region.")
                self.assertLess(abs(region["height_kpc"]), 2)
                screening = next(source["screening_values"] for source in body["sources"]
                                 if "screening_values" in source)
                self.assertEqual(screening["non_single_star"], 0)
                self.assertLess(screening["ruwe"], 1.2)
                self.assertFalse(screening["duplicated_source"])
                self.assertEqual(screening["ipd_frac_multi_peak"], 0)
                spin = next(source for source in body["sources"]
                            if "rotation_period_days" in source["fields"])
                if body["rotation_period_kind"] == "gyrochronology_estimate":
                    self.assertIn("not a calibrated 90% prediction interval", spin["note"])
                    self.assertTrue(spin["limitation_reference"])
                else:
                    self.assertGreater(spin["uncertainties"]["rotation_period_days"], 0)
                    self.assertIn("observed photometric", spin["method"])

    def test_disk_gap_batch_fills_six_bands_with_distinct_sourced_stars(self):
        bodies = [body for body in self.catalog["bodies"]
                  if body.get("catalog_batch") == "disk_gap_2026_10_24"]
        self.assertEqual(len(bodies), 60)
        self.assertEqual(len({body["identifiers"]["gaia_dr3"] for body in bodies}), 60)
        for band in range(1, 7):
            group = [body for body in bodies if body["galactic_region"]["gap_band_index"] == band]
            self.assertEqual(len(group), 10)
            expected = "gaiarvs" if band <= 3 else "lamostl7"
            for body in group:
                self.assertEqual(body["sources"][0]["catalog_table"], f"J/A+A/673/A155/{expected}")
                self.assertFalse(body["sources"][0]["quality_flags"]["OutputFlag"])
                self.assertFalse(body["sources"][0]["quality_flags"]["AgeInOut"])
        measured = [body for body in bodies if body["rotation_period_kind"] == "observed_modulation"]
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0]["identifiers"]["gaia_dr3"], "2993520260320097152")
        self.assertEqual(measured[0]["rotation_period_days"], 5.44)

    def test_saved_config_orbits_recover_catalog_directions_and_distances(self):
        configs = generator.build_configs(self.catalog, ROOT)
        self.assertEqual(len(configs), len(self.catalog["bodies"]))
        for body in self.catalog["bodies"]:
            with self.subTest(body=body["id"]):
                filename = generator.body_name(body["name"]).lower() + ".cfg"
                path = ROOT / "RealGalacticSystems/Systems/MilkyWay/Systems" / filename
                text = configs[path]
                self.assertEqual(path.read_text(encoding="utf-8"), text)
                orbit = {}
                for line in generator.node_contents(text, "Orbit").splitlines():
                    if "=" in line:
                        key, value = (part.strip() for part in line.split("=", 1))
                        orbit[key] = value if key in {"referenceBody", "mode"} else float(value)
                self.assertEqual(orbit["referenceBody"], "Sun")
                self.assertEqual(orbit["mode"], "OFF")
                self.assertEqual(orbit["period"], 3e300)
                point = generator.orbit_position(orbit)
                # Invert the Galactic transform to independently check the encoded sky direction.
                matrix = generator.ICRS_TO_GALACTIC
                x, y, z = [math.fsum(matrix[row][column] * point[row] for row in range(3))
                           for column in range(3)]
                ra = math.degrees(math.atan2(y, x)) % 360
                dec = math.degrees(math.atan2(z, math.hypot(x, y)))
                self.assertAlmostEqual(ra, body["position"]["ra_deg"], places=9)
                self.assertAlmostEqual(dec, body["position"]["dec_deg"], places=9)
                distance = math.hypot(x, y, z) / generator.PARSEC_M
                self.assertTrue(math.isclose(distance, body["position"]["distance_pc"],
                                             rel_tol=1e-14, abs_tol=1e-11))
                self.assertEqual(generator.orbit_position(orbit, 1e12), point)
                recovery = next(line.split("=", 1)[1].split("//", 1)[0].strip()
                                for line in text.splitlines() if "recoveryValue =" in line)
                distance_ly = body["position"]["distance_pc"] * generator.PARSEC_M / generator.LIGHT_YEAR_M
                self.assertEqual(int(recovery), math.ceil(50 + distance_ly))


if __name__ == "__main__":
    unittest.main()
