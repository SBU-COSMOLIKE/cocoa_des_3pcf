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
source = project / 'scripts/measure_full_joint_mpi.py'
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
        args=SimpleNamespace(count=4, start=0, stop=4, output_dir=None,
                             wait_for_ready=True), comm=comm,
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
    # A concurrent range reads only its assigned global IDs and writes to a
    # different directory. The original result remains byte-for-byte intact.
    original = (output / 'map_000002.txt').read_bytes()
    original_partial = np.full(480, np.nan)
    original_partial[:20] = 8.
    original_partial[200:220] = 8.
    np.savetxt(output / 'map_000003.txt', original_partial)
    original_three = (output / 'map_000003.txt').read_bytes()
    shard = folder / 'full_joint_480_shards/2_4'
    shard.mkdir(parents=True)
    shard_clock = iter(range(0, 100000, 11))
    environment.update(output=shard, comm=Comm(),
                       args=SimpleNamespace(count=4, start=2, stop=4,
                                            output_dir=shard,
                                            wait_for_ready=True),
                       time=SimpleNamespace(monotonic=lambda: next(shard_clock),
                                            sleep=lambda _: None))
    exec(code, environment)
    environment['run']()
    assert [n for n, jobs in environment['comm'].assignments] == [3]
    assert environment['comm'].assignments[0][1] == list(range(1, 30))
    assert (output / 'map_000002.txt').read_bytes() == original
    assert (output / 'map_000003.txt').read_bytes() == original_three
    assert not (shard / 'map_000002.txt').exists()
    assert np.loadtxt(shard / 'map_000003.txt')[0] == 8.
    assert (shard / 'COMPLETE.txt').read_text().startswith('2 full joint')
print('PASS: whole-map ownership, delayed arrival, partial resume, saved rows and completion')
