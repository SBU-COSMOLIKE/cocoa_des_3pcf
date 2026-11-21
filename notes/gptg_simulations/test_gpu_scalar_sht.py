"""Compare native Metal scalar transforms with ducc0 and test the adjoint.

The packed harmonic inner product weights m>0 twice, because the negative
orders are supplied by conjugate symmetry. A Euclidean complex dot product
without this factor would test the wrong adjoint.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import time
import numpy as np
import ducc0
import mlx.core as mx
from gpu_scalar_sht import MetalScalarSHT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nside',type=int,default=8)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    lmax=2*args.nside
    geometry=ducc0.healpix.Healpix_Base(args.nside,'RING').sht_info()
    start=time.monotonic()
    sht=MetalScalarSHT(geometry,lmax)
    print('prepared',args.nside,'seconds',time.monotonic()-start,flush=True)
    rng=np.random.default_rng(71)
    na=(lmax+1)*(lmax+2)//2
    alm=(rng.normal(size=na)+1j*rng.normal(size=na)).astype(np.complex64)
    alm[:lmax+1]=alm[:lmax+1].real
    values=rng.normal(size=12*args.nside**2).astype(np.float32)
    synthesis=ducc0.sht.synthesis(alm=alm[None,:],lmax=lmax,spin=0,
                                nthreads=args.threads,**geometry)[0]
    adjoint=ducc0.sht.adjoint_synthesis(map=values[None,:],lmax=lmax,spin=0,
                                      nthreads=args.threads,**geometry)[0]
    agpu=mx.array(alm)
    vgpu=mx.array(values)
    saved=[]
    for repeat in range(3):
        start=time.monotonic()
        candidate=sht.synthesis(agpu)
        mx.eval(candidate)
        forward_time=time.monotonic()-start
        start=time.monotonic()
        candidate_adjoint=sht.adjoint(vgpu)
        mx.eval(candidate_adjoint)
        adjoint_time=time.monotonic()-start
        saved.append((np.array(candidate),np.array(candidate_adjoint)))
        print('repeat',repeat,'GPU forward/adjoint seconds',forward_time,adjoint_time,flush=True)
    out=max((row[0] for row in saved),key=lambda row:np.linalg.norm(row-synthesis))
    back=max((row[1] for row in saved),key=lambda row:np.linalg.norm(row-adjoint))
    weight=np.full(na,2.)
    weight[:lmax+1]=1.
    forward_error=np.linalg.norm(out-synthesis)/np.linalg.norm(synthesis)
    adjoint_error=np.sqrt(np.sum(weight*np.abs(back-adjoint)**2)/np.sum(weight*np.abs(adjoint)**2))
    left=np.dot(out.astype(float),values.astype(float))
    right=np.sum(weight*np.real(np.conj(alm.astype(complex))*back.astype(complex)))
    scale=np.linalg.norm(out)*np.linalg.norm(values)
    dot_error=abs(left-right)/scale
    print('relative RMS forward',forward_error,'adjoint',adjoint_error,'normalized dot error',dot_error,flush=True)
    print('repeat max differences',max(np.max(np.abs(a[0]-saved[0][0])) for a in saved),
          max(np.max(np.abs(a[1]-saved[0][1])) for a in saved))
    if max(forward_error,adjoint_error) > 1.e-5:
        reference64=ducc0.sht.adjoint_synthesis(
            map=values[None,:].astype(np.float64), lmax=lmax, spin=0,
            nthreads=args.threads, **geometry)[0]
        syn64=ducc0.sht.synthesis(alm=alm[None,:].astype(np.complex128),lmax=lmax,spin=0,nthreads=args.threads,**geometry)[0]
        print('ducc32 synthesis vs double',np.linalg.norm(synthesis-syn64)/np.linalg.norm(syn64))
        ring_errors=[]
        for r in range(sht.nr):
            lo=int(sht.ringstart[r]);hi=lo+int(sht.nphi[r])
            ring_errors.append(np.linalg.norm(out[lo:hi]-synthesis[lo:hi]))
        print('worst rings',[(int(r),int(sht.nphi[r]),ring_errors[r]) for r in np.argsort(ring_errors)[-8:]])
        print('ducc32 vs ducc64' ,np.linalg.norm(adjoint-reference64)/np.linalg.norm(reference64))
        order=np.concatenate([np.full(lmax+1-m,m) for m in range(lmax+1)])
        ell=np.concatenate([np.arange(m,lmax+1) for m in range(lmax+1)])
        for index in np.argsort(np.abs(back-adjoint))[-8:]:
            print('worst',int(ell[index]),int(order[index]),back[index],adjoint[index])
        fft_gpu=np.array(sht.ring_fft(vgpu.astype(mx.complex64),inverse=False))
        fft_ref=np.empty_like(fft_gpu)
        for r in range(sht.nr):
            start=int(sht.ringstart[r]);end=start+int(sht.nphi[r])
            fft_ref[start:end]=np.fft.fft(values[start:end])
        print('ring FFT relative error',np.linalg.norm(fft_gpu-fft_ref)/np.linalg.norm(fft_ref))
        np.savez('/tmp/gpu_sht_diagnostic.npz',back=back,reference=adjoint,order=order,ell=ell)
    assert np.isfinite(out).all() and np.isfinite(back).all()
    assert max(forward_error,adjoint_error)<1.e-5
    assert dot_error<1.e-6


if __name__=='__main__':
    main()
