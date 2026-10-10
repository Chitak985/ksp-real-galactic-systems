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


if __name__ == "__main__":
    unittest.main()
