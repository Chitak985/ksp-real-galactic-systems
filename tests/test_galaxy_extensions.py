"""Per-galaxy packaging, safe relocation, and base-mod defaults."""
import copy
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import rebuild_galaxies as galaxy
from rebuild_configs import CatalogError,load_catalog

ROOT=Path(__file__).resolve().parents[1]


class GalaxyExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        shutil.copytree(ROOT/'templates/galaxies',self.root/'templates/galaxies')
        self.catalog=copy.deepcopy(load_catalog(ROOT/'data/galaxies.json'))
        self.catalog['galaxies']=[g for g in self.catalog['galaxies'] if g['id'] in {'ngc_300','ngc_404'}]
        self.catalog['rendering_overrides']={}

    def write(self,selected=None):
        outputs=galaxy.build_configs(self.catalog,self.root,selected)
        galaxy.write_galaxy_configs(outputs,self.catalog,self.root,selected)
        return outputs

    def test_saved_choice_controls_folder_and_all_other_values_default_to_base(self):
        for value,folder in [('Far1','Extension1'),('Far2','Extension2'),('Far3','Extension3'),
                             ('None','Systems'),(None,'Systems'),('Far4','Systems'),
                             ('other','Systems'),('../../escape','Systems'),({},'Systems')]:
            with self.subTest(value=value):
                path=galaxy.config_path({'id':'example','far':value},self.root)
                self.assertEqual(path,self.root/'RealGalacticSystems'/folder/'example.cfg')
        self.assertEqual(galaxy.config_path({'id':'example'},self.root).parent.name,'Systems')
        with self.assertRaises(CatalogError):galaxy.config_path({'id':'../escape'},self.root)

    def test_relocation_and_return_to_base_leave_exactly_one_complete_output(self):
        self.write()
        body=self.catalog['galaxies'][0]
        old=galaxy.config_path(body,self.root)
        body['far']='Far1'
        moved=galaxy.config_path(body,self.root)
        self.assertIn(old,galaxy.obsolete_configs(self.catalog,self.root))
        self.write()
        self.assertFalse(old.exists());self.assertTrue(moved.exists())
        self.assertIn('GalacticDiskDefinition',moved.read_text())
        self.assertEqual(galaxy.obsolete_configs(self.catalog,self.root),{})
        body['far']='unassigned'
        self.write()
        self.assertTrue(old.exists());self.assertFalse(moved.exists())
        self.assertEqual(len(list((self.root/'RealGalacticSystems').rglob('*.cfg'))),2)

    def test_selected_rebuild_does_not_move_unselected_galaxies(self):
        self.write()
        for body in self.catalog['galaxies']:body['far']='Far2'
        selected=self.catalog['galaxies'][0]
        other=self.catalog['galaxies'][1]
        self.write([selected['id']])
        self.assertTrue(galaxy.config_path(selected,self.root).exists())
        self.assertFalse(galaxy.config_path(other,self.root).exists())
        self.assertTrue((self.root/f"RealGalacticSystems/Systems/{other['id']}.cfg").exists())

    def test_unmanaged_old_or_destination_files_are_preserved(self):
        self.write()
        body=self.catalog['galaxies'][0]
        old=galaxy.config_path(body,self.root)
        body['far']='Far3'
        old.write_text('user file\n')
        with self.assertRaises(CatalogError):galaxy.build_configs(self.catalog,self.root)
        self.assertEqual(old.read_text(),'user file\n')
        old.unlink()
        target=galaxy.config_path(body,self.root)
        target.parent.mkdir(parents=True)
        target.write_text('another user file\n')
        with self.assertRaises(CatalogError):galaxy.build_configs(self.catalog,self.root)
        self.assertEqual(target.read_text(),'another user file\n')

    def test_check_and_dry_run_report_old_outputs_without_mutating_them(self):
        self.write()
        for body in self.catalog['galaxies']:body['far']='Far1'
        catalog_path=self.root/'catalog.json'
        import json
        catalog_path.write_text(json.dumps(self.catalog))
        before={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with mock.patch.object(galaxy,'ROOT',self.root),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(galaxy.main(['--catalog',str(catalog_path),'--check']),1)
            self.assertEqual(galaxy.main(['--catalog',str(catalog_path),'--dry-run']),0)
        after={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_all_saved_galaxies_have_one_output_and_manual_hosts_remain_in_systems(self):
        catalog=load_catalog(ROOT/'data/galaxies.json')
        for body in catalog['galaxies']:
            self.assertIn('far',body)
            expected=galaxy.config_path(body,ROOT)
            self.assertTrue(expected.exists(),body['id'])
            copies=[ROOT/f"RealGalacticSystems/{folder}/{body['id']}.cfg" for folder in galaxy.OUTPUT_FOLDERS]
            self.assertEqual([p for p in copies if p.exists()],[expected])
        for body in load_catalog(ROOT/'data/manual_object_positions.json')['objects']:
            if body.get('object_type')=='galaxy':
                self.assertTrue((ROOT/body['path']).exists())
                self.assertTrue(body['path'].startswith('RealGalacticSystems/Systems/'))


if __name__=='__main__':unittest.main()
