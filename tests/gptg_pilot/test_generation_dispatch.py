"""Compare one-process and coordinator generation without computing sky maps.

Execute the actual entry point with a tiny saved-map calculation and fake MPI.
The test covers seed ownership, ordered journal rows and a no-work resume.
The production one-map numerical function is not changed by this test.
"""
import ast
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import argparse
import numpy as np
import yaml
import os

PROJECT = Path(__file__).resolve().parents[2]
source = PROJECT / 'scripts/generate_ensemble_mpi.py'
tree = ast.parse(source.read_text())
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
code = compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), str(source), 'exec')

with tempfile.TemporaryDirectory() as temporary:
    project = Path(temporary)
    (project / 'chains/rafael/maps_cov_512_v2').mkdir(parents=True)
    np.save(project / 'chains/rafael/maps_cov_512_v2/indices_512.npy', np.arange(2))
    pilot = project / 'pilot'
    pilot.mkdir()
    # The frozen geometry and per-pixel noise are required inputs even
    # when this dispatch test replaces the expensive transform itself.
    (pilot / 'calibration.yaml').write_text('lmax: 1\nnside: 512\n')
    np.savez(pilot / 'calibration.npz', cl_target=np.zeros((2,4,4)),
             parameters=np.ones(4), noise_sigma=np.ones(4))
    journals = []
    for size in (1, 2):
        calls = []
        folder = project / f'ranks{size}'

        def generate_one(**kwargs):
            seed = kwargs['seed']
            calls.append(seed)
            np.save(kwargs['path'], np.full((4,2,2), seed, dtype=np.float32))
            return 1., 2., np.full(16, float(seed))

        class Comm:
            def Get_rank(self): return 0
            def Get_size(self): return size
            def Barrier(self): pass
            def send(self, task, dest):
                if task is None: return
                position, (group, number, seed) = task
                result = generate_one(path=folder/group/f'GPTG_DES_{number:06d}.npy', seed=seed)
                self.result = (position, *result, 'cpu_float64_batch1')
            def Iprobe(self, source): return True
            def recv(self, source, status): return self.result

        comm = Comm()
        namespace = dict(argparse=argparse, Path=Path, PROJECT=project, OUTPUT=pilot,
            os=os, np=np, yaml=yaml, hashlib=hashlib, time=time,
            shutil=SimpleNamespace(copy2=shutil.copy2, disk_usage=lambda path: SimpleNamespace(free=10**12)),
            MPI=SimpleNamespace(COMM_WORLD=comm, ANY_SOURCE=-1,
                                Status=lambda: SimpleNamespace(Get_source=lambda: 1)),
            setup=lambda **kw: (np.arange(2), {}, None, None, None, None),
            aperture_weights=lambda **kw: (None,None,None),
            gaussian_spectra=lambda **kw: (np.zeros((2,4,4)),None,None,None),
            generate_one=generate_one, __doc__='Test dispatch')
        exec(code, namespace)
        sys.argv = ['test', '--output',str(folder),'--comparison-count','2',
                    '--validation-count','1','--limit','2','--threads','1']
        namespace['main']()
        assert calls == [341104759,341104760]
        before = (folder/'comparison_measurements.txt').read_bytes()
        calls.clear()
        namespace['main']()
        assert not calls, 'Resume recomputed a completed map'
        assert (folder/'comparison_measurements.txt').read_bytes()==before
        metadata=yaml.safe_load((folder/'ensemble.yaml').read_text())
        assert metadata['generation_workers']==1
        journals.append(before)
    assert journals[0]==journals[1]
print('PASS: single-process and coordinator paths match seeds/journals; resume skips completed maps.')
