"""Exercise the actual coordinator with fake MPI messages, never TreeCorr.

Map zero is complete; map one has one completed task. Map three arrives
later. Every fake task returns deterministic entries, testing checkpoint
preservation and that workers are not released at intermediate messages.
"""
import ast
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np

project = Path(__file__).resolve().parents[2]
source = project / 'notes/gptg_simulations/measure_full_joint_mpi.py'
tree = ast.parse(source.read_text())
branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
              and any(isinstance(a, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == 'pending' for t in a.targets)
                      for a in n.body))
start = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'pending' for t in n.targets))
function = ast.FunctionDef(name='run', args=ast.arguments(posonlyargs=[], args=[],
    kwonlyargs=[], kw_defaults=[], defaults=[]), body=branch.body[start:], decorator_list=[])
code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
               str(source), 'exec')

class Status:
    def Get_source(self):
        return self.rank

with TemporaryDirectory() as temporary:
    folder = Path(temporary)
    output = folder / 'full_joint_480'
    output.mkdir()
    for group in ['comparison', 'validation']:
        (folder / group).mkdir()
    np.savetxt(output / 'map_000000.txt', np.full(480, 7.))
    partial = np.full(480, np.nan)
    partial[:20] = 9.
    partial[200:220] = 9.
    np.savetxt(output / 'map_000001.txt', partial)
    for number in [1, 2]:
        (folder / 'comparison' / f'GPTG_DES_{number:06d}.npy.ready').touch()
    (output / 'COMPLETE.txt').write_text('old smaller target')

    class Comm:
        def __init__(self):
            self.queue = deque()
            self.busy = set()
            self.assignments = []
        def send(self, task, dest):
            if task is None:
                assert dest not in self.busy
                return
            assert dest not in self.busy, 'worker released before finishing its map'
            self.busy.add(dest)
            number, jobs = task
            self.assignments.append((number, jobs))
            for j in jobs:
                self.queue.append((dest, number, j, j == jobs[-1]))
            (folder / 'comparison/GPTG_DES_000003.npy.ready').touch()
        def Iprobe(self, source):
            return bool(self.queue)
        def recv(self, source, status):
            rank, number, job, last = self.queue.popleft()
            status.rank = rank
            if last:
                self.busy.remove(rank)
            return number, job, np.full(40 if job < 10 else 4, number+1.), 1.

    clock = iter(range(0, 100000, 11))
    comm = Comm()
    environment = dict(np=np, deque=deque, folder=folder, output=output, size=3,
        args=SimpleNamespace(count=4, wait_for_ready=True), comm=comm,
        MPI=SimpleNamespace(ANY_SOURCE=-1, Status=Status),
        time=SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None),
        print=lambda *a, **k: None)
    exec(code, environment)
    environment['run']()
    assert [n for n, jobs in comm.assignments] == [1, 2, 3]
    assert comm.assignments[0][1] == list(range(1, 30))
    assert all(np.isfinite(np.loadtxt(output/f'map_{n:06d}.txt')).all() for n in range(4))
    np.testing.assert_array_equal(np.loadtxt(output/'map_000000.txt'), np.full(480,7.))
    assert np.loadtxt(output/'map_000001.txt')[0] == 9.
    assert (output/'COMPLETE.before_4.txt').exists()
    assert (output/'COMPLETE.txt').read_text().startswith('4 full joint')
print('PASS: whole-map ownership, delayed arrival, partial resume, saved rows and completion')
