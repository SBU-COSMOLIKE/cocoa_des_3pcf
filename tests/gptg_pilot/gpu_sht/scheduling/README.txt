48 fresh maps per configuration; NSIDE512/lmax1024; BLAS=1.
Four generation workers x three native CPU threads, plus coordinator.
Production stopped during timings. The worker-count reverse trial briefly
coincided with the forward numerical checker; the forward trial had no
such checker. Both disfavor two GPU workers. Pipeline timings had no
concurrent numerical checker and both favor overlap by about 6.7%.

Reproduce using benchmark_gpu_sht_mpi.py --compare-gpu-workers or
--compare-pipeline, --count 48, and a fresh --output under chains/.
Repeat with --reverse and another fresh output. Check results with
check_gpu_worker_comparison.py, adding --pipeline for overlap comparisons.
The reference is the first 48 original production CPU maps (same seeds).
Noiseless comparison: test_gpu_map_generation.py --count 2 --threads 3 --pipeline.

Pipeline common GPU seeds were pixel-identical to sequential GPU seeds.
This tests scheduling and numerical agreement, not physical calibration
or full TreeCorr statistics. Timings include process startup and map saves.
