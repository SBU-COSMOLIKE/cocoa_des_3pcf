"""Validate full GPU-scalar/CPU-shear maps against the unchanged CPU generator."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import time
from pathlib import Path
import numpy as np
import yaml
from generate_maps import PROJECT,setup,gaussian_spectra,aperture_weights
from generate_ensemble import generate_one
from gpu_scalar_sht import MetalScalarSHT
from gpu_map_generation import generate_gpu_map


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--pipeline',action='store_true',
                        help='Validate overlap of CPU shear and next-bin GPU work.')
    args=parser.parse_args()
    output=PROJECT/'chains/gpu_sht_maps'
    output.mkdir(exist_ok=True)
    source=PROJECT/'chains/gptg_ensemble_15000'
    record=yaml.safe_load((source/'calibration.yaml').read_text())
    with np.load(source/'calibration.npz') as archive:
        data=dict(archive)
    data['noise_sigma']=np.zeros_like(data['noise_sigma'])
    indices,geometry,ell,factor,nodes,weights=setup(lmax=record['lmax'])
    first,second,centers=aperture_weights(indices=indices)
    latent,variance,repair,diagonal=gaussian_spectra(
        cl=data['cl_target'],parameters=data['parameters'],nodes=nodes,weights=weights,
    )
    eigenvalues,eigenvectors=np.linalg.eigh(latent)
    roots=eigenvectors*np.sqrt(np.maximum(eigenvalues,0.))[:,None,:]
    sht=MetalScalarSHT(geometry,record['lmax'])
    for number in range(args.count):
        seed=641104759+number
        path=output/f'cpu_{number:03d}.npy'
        path.unlink(missing_ok=True)
        start=time.monotonic()
        generate_one(path=path,seed=seed,record=record,data=data,indices=indices,
                     geometry=geometry,ell=ell,factor=factor,nodes=nodes,weights=weights,
                     roots=roots,variance=variance,first=first,second=second,workers=args.threads)
        cpu_seconds=time.monotonic()-start
        start=time.monotonic()
        candidate,diagnostics=generate_gpu_map(
            sht=sht,seed=seed,record=record,data=data,roots=roots,variance=variance,
            ell=ell,factor=factor,indices=indices,geometry=geometry,nodes=nodes,
            weights=weights,threads=args.threads,pipeline=args.pipeline,
        )
        gpu_seconds=time.monotonic()-start
        np.save(output/f'gpu_{number:03d}.npy',candidate)
        np.savetxt(output/f'stages_{number:03d}.txt',diagnostics,
                   header='source synthesis point solve shear iterations true_normal_residual')
        reference=np.load(path).astype(float)
        candidate=candidate.astype(float)
        error=np.max(np.sqrt(np.mean((candidate-reference)**2,axis=(1,2))
                             /np.mean(reference**2,axis=(1,2))))
        aperture=first@reference[:,0].T+second@reference[:,1].T
        trial=first@candidate[:,0].T+second@candidate[:,1].T
        moments=[]
        for power in range(2,7):
            moments.append(np.max(np.abs(np.mean(trial**power-aperture**power,axis=0))
                                  /np.mean(np.abs(aperture)**power,axis=0)))
        print('map',number,'CPU seconds',cpu_seconds,'GPU seconds',gpu_seconds,
              'signal error',error,'moment errors',moments,flush=True)
        assert np.isfinite(candidate).all()
        assert max(error,*moments)<1.e-4


if __name__=='__main__':
    main()
