"""Distance routing for galaxy toggle groups, including manual host bodies."""
import json
import math
from pathlib import Path
import re
import unittest

import rebuild_galaxies as galaxies
import rebuild_manual_orbits as manual
from rebuild_configs import CatalogError, node_contents

ROOT=Path(__file__).resolve().parents[1]


class GalaxyGroupTests(unittest.TestCase):
    def test_distance_boundaries(self):
        for boundary,below,above in ((100,"None","Far1"),(125,"Far1","Far2"),(150,"Far2","Far3")):
            with self.subTest(boundary=boundary):
                self.assertEqual(galaxies.distance_group(math.nextafter(boundary,-math.inf)),below)
                self.assertEqual(galaxies.distance_group(boundary),below)
                self.assertEqual(galaxies.distance_group(math.nextafter(boundary,math.inf)),above)
        self.assertEqual(galaxies.distance_group(0),"None")
        self.assertEqual(galaxies.distance_group(10400),"Far3")

    def test_invalid_distances_are_rejected(self):
        for value in (-1,math.nan,math.inf,-math.inf,None):
            with self.subTest(value=value),self.assertRaises(CatalogError):
                galaxies.distance_group(value)

    def test_shipped_tags_cover_galaxies_and_preserve_other_objects(self):
        def check(path,expected):
            body=node_contents(path.read_text(encoding="utf-8-sig"),"Body")
            # Count only direct Body fields, not fields in Properties or disks.
            depth=0;values=[]
            for line in body.splitlines():
                if depth==0:
                    match=re.match(r"\s*RGSType\s*=\s*(\w+)",line)
                    if match:values.append(match[1])
                depth+=line.count("{")-line.count("}")
            self.assertEqual(values,[] if expected is None else [expected],str(path))
        catalog=json.loads((ROOT/"data/galaxies.json").read_text())
        for record in catalog["galaxies"]:
            d=galaxies.distance_measures(record,catalog["settings"])["placement_mpc"]
            check(ROOT/f"RealGalacticSystems/Systems/{record['id']}.cfg",galaxies.distance_group(d))
        catalog=json.loads((ROOT/"data/manual_object_positions.json").read_text())
        marked=set()
        for record in catalog["objects"]:
            expected=None
            if record.get("object_type")=="galaxy":
                marked.add(record["body_name"])
                expected="None"
            check(ROOT/record["path"],expected)
        self.assertEqual(marked,{"SagittariusA","AndromedaA","TriangulumBarycenter",
            "LargeMagellanicCloud","SmallMagellanicCloud","NGC5866","GN-z11","JADES-GS-Z14-0"})

    def test_far_disks_silence_missing_body_errors(self):
        catalog=json.loads((ROOT/"data/galaxies.json").read_text())
        for record in catalog["galaxies"]:
            text=(ROOT/f"RealGalacticSystems/Systems/{record['id']}.cfg").read_text()
            group=re.search(r"(?m)^\s*RGSType\s*=\s*(\w+)",node_contents(text,"Body"))[1]
            disk=node_contents(text,"GalacticDiskDefinition")
            fields=re.findall(r"(?m)^\s*silenceNoBodyError\s*=\s*(\w+)",disk)
            self.assertEqual(fields,["True"] if group in {"Far1","Far2","Far3","Far4","Far5"} else [],record["id"])

    def test_manual_tag_only_updates_body_level_and_is_idempotent(self):
        original="Body\r\n{\r\n  name = Example\r\n  Properties\r\n  {\r\n    RGSType = Nested\r\n  }\r\n}\r\n"
        tagged=manual.set_galaxy_group(original,"Far1")
        self.assertEqual(tagged.replace("  RGSType = Far1\r\n","",1),original)
        self.assertEqual(manual.set_galaxy_group(tagged,"Far1"),tagged)
        updated=manual.set_galaxy_group(tagged.replace("RGSType = Far1","RGSType = Far1 // keep comment",1),"Far2")
        self.assertIn("  RGSType = Far2 // keep comment\r\n",updated)
        self.assertIn("    RGSType = Nested\r\n",updated)
        self.assertNotIn("\n",updated.replace("\r\n",""))
        with self.assertRaises(CatalogError):
            manual.set_galaxy_group(tagged.replace("  name = Example","  RGSType = Far2\r\n  name = Example"),"Far3")


if __name__=="__main__":unittest.main()
