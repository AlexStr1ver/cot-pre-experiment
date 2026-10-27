"""Recovery routing tests: no install, network or GPU work is performed."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_environment as recovery


HEALTHY = {'passed':True,'probe_exit_status':0,'dependencies_ok':True,'package_versions_match':True,
           'python_version_compatible':True,'cuda_ok':True,'hardware_compatible':True}
BROKEN = {'passed':False,'probe_exit_status':1,'dependencies_ok':False,'package_versions_match':False,
          'python_version_compatible':True,'cuda_ok':None,'hardware_compatible':None}


class PreparationTests(unittest.TestCase):
    def test_zero_exit_without_passed_json_is_not_a_healthy_probe(self):
        self.assertFalse(recovery.usable({'probe_exit_status':0}))
        self.assertFalse(recovery.usable({'probe_exit_status':0,'passed':False}))
        self.assertTrue(recovery.usable(HEALTHY))

    def fixture(self, directory):
        root=Path(directory)/'root with spaces'; root.mkdir()
        package=Path(directory)/'package'; package.mkdir()
        run=root/'new run'; run.mkdir()
        previous=root/'completed main'; (previous/'results').mkdir(parents=True)
        (previous/'results/run_config.json').write_text(json.dumps({'mode':'main',
            'python':f'{sys.version_info.major}.{sys.version_info.minor}.0',
            'hardware':{'torch':'2.6.0+cu124'},'transformers':'4.51.3'}))
        (previous/'results/summary.json').write_text(json.dumps({'completed':True,'numerical_audit_passed':True}))
        return root,package,run,previous

    def test_healthy_isolated_python_avoids_install_despite_normal_shadowing(self):
        with tempfile.TemporaryDirectory() as d:
            root,package,run,previous=self.fixture(d)
            with patch.object(recovery,'run_probe',side_effect=[BROKEN,HEALTHY]), \
                 patch.object(recovery,'install_new_environment') as install,patch.object(recovery.subprocess,'run'):
                selected=recovery.prepare(root,package,run,previous)
                install.assert_not_called()
            state=recovery.read(run/'environment/preparation.json')
            self.assertTrue(state['completed']);self.assertEqual(state['selection'],'isolated_import_recovery')
            self.assertEqual(selected,root/'venvs/infty-official/bin/python')
            self.assertFalse((run/'results').exists())

    def test_broken_packages_build_new_environment_and_record_ready_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root,package,run,previous=self.fixture(d)
            fixed=root/'venvs/new repair/bin/python';fixed.parent.mkdir(parents=True)
            with patch.object(recovery,'run_probe',side_effect=[BROKEN,BROKEN,HEALTHY]), \
                 patch.object(recovery,'install_new_environment',return_value=fixed) as install, \
                 patch.object(recovery.subprocess,'run'):
                self.assertEqual(recovery.prepare(root,package,run,previous),fixed)
                install.assert_called_once()
            marker=recovery.read(root/'venvs/infty-paired-v3_1-ready.json')
            self.assertEqual(marker['python'],str(fixed))
            self.assertFalse((root/'venvs/infty-official').exists())

    def test_hardware_or_python_mismatch_blocks_install(self):
        for field in ['cuda_ok','hardware_compatible','python_version_compatible']:
            bad={**HEALTHY,field:False,'probe_exit_status':1}
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                root,package,run,previous=self.fixture(d)
                with patch.object(recovery,'run_probe',return_value=bad),patch.object(recovery,'install_new_environment') as install:
                    with self.assertRaisesRegex(RuntimeError,'mismatch'):
                        recovery.prepare(root,package,run,previous)
                    install.assert_not_called()
                self.assertEqual(recovery.read(run/'environment/preparation.json')['stage'],'failed')
                self.assertFalse((run/'environment/selected_python.txt').exists())
        physical={**BROKEN,'mismatches':[{'kind':'hardware','name':'name','actual':'different GPU'}]}
        self.assertTrue(recovery.cannot_fix_with_dependencies(physical))
        self.assertTrue(recovery.cannot_fix_with_dependencies({**HEALTHY,'cuda_ok':None,'hardware_compatible':None}))

    def test_optional_inventory_failure_does_not_block_healthy_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root,package,run,previous=self.fixture(d)
            with patch.object(recovery,'run_probe',return_value=HEALTHY), \
                 patch.object(recovery.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='pip unavailable')):
                recovery.prepare(root,package,run,previous)
            self.assertTrue(recovery.read(run/'environment/preparation.json')['completed'])
            self.assertFalse(recovery.read(run/'environment/inventory_status.json')['completed'])

    def test_existing_repair_marker_reused_instead_of_install(self):
        with tempfile.TemporaryDirectory() as d:
            root,package,run,previous=self.fixture(d)
            cached=root/'venvs/infty-paired-v3_1_cached/bin/python';cached.parent.mkdir(parents=True)
            recovery.dump(root/'venvs/infty-paired-v3_1-ready.json',{'python':str(cached)})
            with patch.object(recovery,'run_probe',side_effect=[BROKEN,BROKEN,HEALTHY,HEALTHY]), \
                 patch.object(recovery,'install_new_environment') as install,patch.object(recovery.subprocess,'run'):
                self.assertEqual(recovery.prepare(root,package,run,previous),cached)
                install.assert_not_called()

    def test_invalid_or_incomplete_source_blocks_any_install(self):
        with tempfile.TemporaryDirectory() as d:
            root,package,run,previous=self.fixture(d)
            (previous/'results/summary.json').write_text('{"completed":false,"numerical_audit_passed":true}')
            with patch.object(recovery,'run_probe') as probe,patch.object(recovery,'install_new_environment') as install:
                with self.assertRaisesRegex(RuntimeError,'completed main'):
                    recovery.prepare(root,package,run,previous)
                probe.assert_not_called();install.assert_not_called()

    def test_installer_commands_are_pinned_new_venv_outside_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root,package,run,previous=self.fixture(d);folder=run/'environment';folder.mkdir()
            free=type('Disk',(),{'free':30*2**30})()
            with patch.object(recovery.shutil,'disk_usage',return_value=free),patch.object(recovery.subprocess,'run') as command:
                selected=recovery.install_new_environment(root,package,folder,previous,{})
            self.assertTrue(selected.parent.parent.is_relative_to(root/'venvs'))
            self.assertFalse(selected.is_relative_to(run))
            calls=[c.args[0]for c in command.call_args_list]
            self.assertEqual(len(calls),4)
            self.assertIn('torch==2.6.0+cu124',calls[1])
            self.assertIn('https://download.pytorch.org/whl/cu124',calls[1])
            self.assertIn(str(package/'requirements-recovery.txt'),calls[2])
            self.assertNotIn('--system-site-packages',calls[0])
            self.assertNotIn(str(root/'venvs/infty-official/bin/python'),sum(calls,[]))

    def test_installer_failure_has_no_selected_python_or_ready_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root,package,run,previous=self.fixture(d)
            with patch.object(recovery,'run_probe',return_value=BROKEN), \
                 patch.object(recovery,'install_new_environment',side_effect=RuntimeError('download failed')):
                with self.assertRaisesRegex(RuntimeError,'download failed'):
                    recovery.prepare(root,package,run,previous)
            state=recovery.read(run/'environment/preparation.json')
            self.assertFalse(state['completed']);self.assertIn('download failed',state['error'])
            self.assertFalse((run/'environment/selected_python.txt').exists())
            self.assertFalse((root/'venvs/infty-paired-v3_1-ready.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
