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
    kwonlyargs=[], kw_defaults=[], defaults=[]), body=branch.body, decorator_list=[])
code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
               str(source), 'exec')

# Load only the revision guard and constants; importing the MPI runner would
# initialize dependencies which the coordinator test does not need.
definitions = [n for n in tree.body if
    (isinstance(n, ast.FunctionDef) and n.name == 'check_measurement_revision') or
    (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
     t.id in ('MEASUREMENT_REVISION', 'VECTOR_HEADER') for t in n.targets))]
revision_environment = {}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'),
     revision_environment)
header = revision_environment['VECTOR_HEADER']
metadata = (f"measurement_revision={revision_environment['MEASUREMENT_REVISION']}\n"
            'smoke_pixels=0; smoke outputs are not science vectors.\n')

class Status:
    def Get_source(self):
        return self.rank

with TemporaryDirectory() as temporary:
    folder = Path(temporary)
    output = folder / 'full_joint_480'
    output.mkdir()
    for group in ['comparison', 'validation']:
        (folder / group).mkdir()
    (output / 'measurement.txt').write_text(metadata)
    np.savetxt(output / 'map_000000.txt', np.full(480, 7.), header=header)
    partial = np.full(480, np.nan)
    partial[:20] = 9.
    partial[200:220] = 9.
    np.savetxt(output / 'map_000001.txt', partial, header=header)
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
        args=SimpleNamespace(count=4, start=0, stop=4, wait_for_ready=True, smoke_pixels=0), comm=comm,
        MPI=SimpleNamespace(ANY_SOURCE=-1, Status=Status),
        time=SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None),
        print=lambda *a, **k: None,
        treecorr=SimpleNamespace(__version__='test'), first_side=11,
        side_edges=np.geomspace(.5, 80, 21), **revision_environment)
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
    shard = folder / 'full_joint_480_shards/2_4'
    shard.mkdir(parents=True)
    shard_clock = iter(range(0, 100000, 11))
    environment.update(output=shard, comm=Comm(),
                       args=SimpleNamespace(count=4, start=2, stop=4,
                                            wait_for_ready=True, smoke_pixels=0),
                       time=SimpleNamespace(monotonic=lambda: next(shard_clock),
                                            sleep=lambda _: None))
    exec(code, environment)
    environment['run']()
    assert [n for n, jobs in environment['comm'].assignments] == [2, 3]
    assert (output / 'map_000002.txt').read_bytes() == original
    assert (shard / 'COMPLETE.txt').read_text().startswith('2 full joint')
    # Legacy metadata or a legacy vector header must fail before metadata is
    # overwritten and before any worker receives a map. Never migrate rows.
    for old_metadata, old_header in [('legacy definition\n', header),
                                     (metadata, 'old vector'),
                                     (metadata.replace('smoke_pixels=0',
                                                       'smoke_pixels=24'), header)]:
        legacy = folder / 'legacy'
        legacy.mkdir(exist_ok=True)
        (legacy / 'measurement.txt').write_text(old_metadata)
        np.savetxt(legacy / 'map_000000.txt', np.full(480, 7.), header=old_header)
        before = (legacy / 'measurement.txt').read_bytes()
        environment.update(output=legacy, comm=Comm())
        exec(code, environment)
        try:
            environment['run']()
        except ValueError as error:
            assert ('Legacy measurement' in str(error)
                    or 'smoke_pixels mismatch' in str(error))
        else:
            raise AssertionError('Legacy checkpoint was accepted')
        assert (legacy / 'measurement.txt').read_bytes() == before
        assert not environment['comm'].assignments
    # Missing metadata does not turn existing vectors into a new run.
    (legacy / 'measurement.txt').unlink()
    try:
        revision_environment['check_measurement_revision'](legacy)
    except ValueError:
        pass
    else:
        raise AssertionError('Unlabelled checkpoint was accepted')
    # Exercise the actual merge branch too: refuse every legacy shard before
    # changing target metadata or copying any valid earlier shard's vectors.
    merge_branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                        and isinstance(n.test, ast.Attribute)
                        and n.test.attr == 'merge_shards')
    merge_function = ast.FunctionDef(name='merge', args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=merge_branch.body, decorator_list=[])
    merge_code = compile(ast.fix_missing_locations(ast.Module(
        body=[merge_function], type_ignores=[])), str(source), 'exec')
    merge_root = folder / 'merge_case'
    target = merge_root / 'full_joint_480'
    good = merge_root / 'full_joint_480_shards/0_1'
    bad = merge_root / 'full_joint_480_shards/1_2'
    for directory in (target, good, bad):
        directory.mkdir(parents=True)
        (directory / 'measurement.txt').write_text(metadata)
    np.savetxt(good / 'map_000000.txt', np.full(480, 1.), header=header)
    np.savetxt(bad / 'map_000001.txt', np.full(480, 2.), header='old vector')
    environment.update(args=SimpleNamespace(maps_dir=merge_root))
    exec(merge_code, environment)
    try:
        environment['merge']()
    except ValueError as error:
        assert 'Legacy measurement' in str(error)
    else:
        raise AssertionError('Legacy shard was merged')
    assert not list(target.glob('map_*.txt'))
    assert (target / 'measurement.txt').read_text() == metadata
    np.savetxt(bad / 'map_000001.txt', np.full(480, 2.), header=header)
    (bad / 'measurement.txt').write_text(metadata.replace('smoke_pixels=0',
                                                        'smoke_pixels=24'))
    try:
        environment['merge']()
    except ValueError as error:
        assert 'smoke_pixels mismatch' in str(error)
    else:
        raise AssertionError('Different pixel subsets were merged')
    assert not list(target.glob('map_*.txt'))
    (bad / 'measurement.txt').write_text(metadata)
    environment['merge']()
    assert np.all(np.loadtxt(target / 'map_000000.txt') == 1.)
    assert np.all(np.loadtxt(target / 'map_000001.txt') == 2.)
    revision_environment['check_measurement_revision'](target)
print('PASS: revision refusal, whole-map ownership, delayed arrival, partial resume, saved rows and completion')
