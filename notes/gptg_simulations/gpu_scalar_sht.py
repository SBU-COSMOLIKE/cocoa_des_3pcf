"""Experimental spin-zero HEALPix synthesis/adjoint using MLX and Metal.

Longitude uses GPU FFTs at each distinct HEALPix ring length. Latitude uses
normalized associated-Legendre recurrences in custom Metal kernels. A pair
of floats carries recurrence rounding corrections; binary exponents prevent
underflow of the diagonal seed before it grows to a significant harmonic.
The output arrays are float32/complex64. This is not a spin-two transform.
How a scalar sky passes through this module:
1. The latitude recurrence forms b[r,m] = sum_l a[l,m] lambda[l,m](theta[r]).
2. Ring phases and positive/negative-frequency aliases fill longitude bins.
3. Exact-length ring Fourier sums turn those bins into real sky pixels.
4. The adjoint reverses these operations and sums pixels into coefficients.
5. Conjugate gradients repeat synthesis and adjoint to solve the finite
   least-squares inverse. The adjoint alone is not that inverse.

Geometry, recurrence factors and FFT chirps are prepared once per worker.
They depend on the ring geometry and lmax, never on the random realization.
All scalar arrays stay on device during a solve; only convergence scalars
are read by Python. Final spin-two shear synthesis belongs to ducc0 in
gpu_map_generation.py. The companion spherical_transforms.tex derives
these steps, including the packed-coefficient scalar product.

The optional production route passed the recorded numerical gates at
NSIDE512/lmax1024. Other resolutions require their own agreement checks.
"""
import numpy as np
from scipy.special import gammaln
import mlx.core as mx

# Error-free addition and FMA-based multiplication retain a low correction.
# Explicit FMA products retain their rounding residual. Disable implicit
# contraction so the addition correction keeps its stated operation order.
HEADER = r'''
#pragma clang fp contract(off)
inline float2 add2(float2 a, float2 b) {
    float s = a.x+b.x;
    float v = s-a.x;
    float e = ((a.x-(s-v))+(b.x-v))+a.y+b.y;
    float h = s+e;
    return float2(h, e-(h-s));
}
inline float2 mul2(float2 a, float2 b) {
    float p = a.x*b.x;
    float e = fma(a.x,b.x,-p)+a.x*b.y+a.y*b.x;
    float h = p+e;
    return float2(h,e-(h-p));
}
'''
FORWARD = r'''
uint work_index = thread_position_in_grid.x;
if (work_index >= NR*NM) return;
// Adjacent lanes use the same m and different rings, sharing coefficient
// reads through the GPU cache throughout the latitude recurrence.
uint m = work_index/NR, r = work_index%NR, index=r*NM+m;
uint start = starts[m];
float2 q = float2(seed[2*index],seed[2*index+1]);
float2 previous = float2(0.0f);
float2 x = float2(costheta[2*r],costheta[2*r+1]);
int exponent = exponents[index];
float2 sumr = float2(0.0f), sumi = float2(0.0f);
for (uint l=m; l<NM; ++l) {
    uint k = start+l;
    if (l>m) {
        float2 a = float2(ca[2*k],ca[2*k+1]);
        float2 b = float2(cb[2*k],cb[2*k+1]);
        float2 next = add2(mul2(mul2(a,x),q),-mul2(b,previous));
        previous=q; q=next;
        if (abs(q.x)>1.e20f) {
            q *= 0x1p-64f; previous *= 0x1p-64f; exponent+=64;
        }
    }
    float2 value = float2(ldexp(q.x,exponent),ldexp(q.y,exponent));
    sumr=add2(sumr,mul2(value,float2(alm[2*k],0.0f)));
    sumi=add2(sumi,mul2(value,float2(alm[2*k+1],0.0f)));
}
out[2*index]=sumr.x+sumr.y;
out[2*index+1]=sumi.x+sumi.y;
'''
ADJOINT = r'''
// One SIMD group sums 32 latitude rings for each m and each l. Partial
// coefficients are reduced on the GPU afterward; no floating atomics.
uint lane=thread_index_in_simdgroup;
uint block=threadgroup_position_in_grid.x;
uint m=block%NM, tile=block/NM, r=tile*32+lane;
uint index=r*NM+m, start=starts[m];
bool valid=r<NR;
float2 q=valid ? float2(seed[2*index],seed[2*index+1]):float2(0.0f);
float2 previous=float2(0.0f);
float2 x=valid ? float2(costheta[2*r],costheta[2*r+1]):float2(0.0f);
int exponent=valid ? exponents[index]:0;
float fr=valid ? rings[2*index]:0.0f;
float fi=valid ? rings[2*index+1]:0.0f;
for (uint l=m; l<NM; ++l) {
    uint k=start+l;
    if(l>m) {
        float2 a=float2(ca[2*k],ca[2*k+1]);
        float2 b=float2(cb[2*k],cb[2*k+1]);
        float2 next=add2(mul2(mul2(a,x),q),-mul2(b,previous));
        previous=q; q=next;
        if(abs(q.x)>1.e20f) {
            q*=0x1p-64f; previous*=0x1p-64f; exponent+=64;
        }
    }
    float value=ldexp(q.x,exponent)+ldexp(q.y,exponent);
    float sr=simd_sum(value*fr), si=simd_sum(value*fi);
    if(lane==0) {out[2*(tile*NA+k)]=sr;out[2*(tile*NA+k)+1]=si;}
}
'''
ALIAS = r'''
uint pixel=thread_position_in_grid.x;
if(pixel>=NP) return;
uint r=ring_of_pixel[pixel], n=nphi[r], j=pixel-ringstart[r];
float2 sum=float2(0.0f);
// Both signs of m contribute. When they alias to the same FFT bin,
// both are needed, except that m=0 must be counted only once.
for(uint m=j;m<NM;m+=n) {
    uint k=r*NM+m;
    float ar=rings[2*k], ai=rings[2*k+1];
    float c=phase[2*k], s=phase[2*k+1];
    sum+=float2(ar*c-ai*s,ar*s+ai*c);
}
for(uint m=(j==0 ? n:n-j);m<NM;m+=n) {
    uint k=r*NM+m;
    float ar=rings[2*k], ai=rings[2*k+1];
    float c=phase[2*k], s=phase[2*k+1];
    sum+=float2(ar*c-ai*s,-ar*s-ai*c);
}
out[2*pixel]=sum.x;out[2*pixel+1]=sum.y;
'''
UNALIAS = r'''
uint index=thread_position_in_grid.x;
if(index>=NR*NM) return;
uint r=index/NM,m=index%NM,k=ringstart[r]+m%nphi[r];
float ar=fftmap[2*k],ai=fftmap[2*k+1];
float c=phase[2*index],s=phase[2*index+1];
out[2*index]=ar*c+ai*s;
out[2*index+1]=ai*c-ar*s;
'''


def split(values):
    """Encode a double as a leading float plus a small float correction."""
    high = np.asarray(values, dtype=np.float32)
    return mx.array(np.stack((high, np.asarray(values)-high), axis=-1).astype(np.float32))


class MetalScalarSHT:
    """Cache geometry and recurrence constants; keep transformed arrays on GPU."""

    def __init__(self, geometry, lmax):
        self.lmax = lmax
        self.nm = lmax+1
        self.nr = len(geometry['theta'])
        self.nphi = geometry['nphi'].astype(np.int32)
        self.ringstart = geometry['ringstart'].astype(np.int32)
        self.npix = int(self.nphi.sum())
        # Packed storage has m=0 coefficients first, then m=1, and so on.
        # There are lmax-m+1 degrees for each order m. Summing those lengths
        # gives (lmax+1)*(lmax+2)/2 independent complex slots; m=0 is real.
        self.na = (lmax+1)*(lmax+2)//2
        self.tiles = (self.nr+31)//32
        m = np.arange(self.nm)
        starts = m*(2*lmax+1-m)//2
        self.starts = mx.array(starts.astype(np.int32))
        self.costheta = split(np.cos(geometry['theta']))
        # Compute log|lambda_mm| on the CPU before splitting the seed into
        # a mantissa and a base-two exponent. Near a pole, (sin theta)^m can
        # underflow even though later degrees have significant amplitudes.
        # Retaining the exponent lets the recurrence reach those degrees.
        logseed = (.5*np.log((2*m+1)/(4*np.pi))+.5*gammaln(2*m+1)
                   -gammaln(m+1)-m*np.log(2.)
                   +np.log(np.sin(geometry['theta']))[:,None]*m)
        exponent = np.floor(logseed/np.log(2.)).astype(np.int32)
        self.seed = split(np.exp(logseed-exponent*np.log(2.))*(-1.)**m)
        self.exponents = mx.array(exponent)
        # Normalized Legendre recurrence:
        # lambda_lm = (x*lambda_(l-1,m) - A_(l-1,m)*lambda_(l-2,m))/A_lm.
        # Store ca=1/A_lm and cb=A_(l-1,m)/A_lm once, in packed order.
        # Two-float representations reduce rounding drift along l without
        # requiring native float64 operations on the Apple GPU.
        ca = np.zeros(self.na)
        cb = np.zeros(self.na)
        for order in m:
            ell = np.arange(order+1, self.nm, dtype=float)
            a = np.sqrt((ell**2-order**2)/(4*ell**2-1))
            previous = np.zeros_like(a)
            valid = ell > 1
            previous[valid] = np.sqrt(((ell[valid]-1)**2-order**2)
                                     /(4*(ell[valid]-1)**2-1))
            ca[starts[order]+ell.astype(int)] = 1/a
            cb[starts[order]+ell.astype(int)] = previous/a
        self.ca, self.cb = split(ca), split(cb)
        # HEALPix rings do not all start at longitude zero. This phase
        # rotates each ring's Fourier wave to its actual first pixel.
        angle = geometry['phi0'][:, None]*m
        self.phase = mx.array(np.stack((np.cos(angle), np.sin(angle)), axis=-1).astype(np.float32))
        self.ngpu = mx.array(self.nphi)
        self.sgpu = mx.array(self.ringstart)
        self.rgpu = mx.array(np.repeat(np.arange(self.nr, dtype=np.int32), self.nphi))
        # Bluestein converts different ring lengths to a few power-of-two
        # convolution lengths. Grouping these convolutions avoids hundreds
        # of separate GPU FFT launches for the polar rings.
        self.fft_groups = []
        lengths = 1 << np.ceil(np.log2(2*self.nphi-1)).astype(int)
        equator = self.nphi == np.max(self.nphi)
        lengths[equator] = 0
        for length in np.unique(lengths):
            rings = np.flatnonzero(lengths == length)
            n = self.nphi[rings, None]
            size = int(length) if length else int(n[0,0])
            j = np.arange(size)[None,:]
            index = self.ringstart[rings,None]+j
            index = np.where(j<n,index,0).astype(np.int32)
            if length:
                chirp = np.where(j<n,np.exp(-1j*np.pi*j*j/n),0).astype(np.complex64)
                signed = np.where(j<n,j,j-size)
                valid = (j<n)|(j>=size-n+1)
                convolution = np.where(valid,np.exp(1j*np.pi*signed*signed/n),0).astype(np.complex64)
                kernel = mx.fft.fft(mx.array(convolution),axis=-1)
                mx.eval(kernel)
                chirp = mx.array(chirp)
            else:
                chirp, kernel = None, None
            self.fft_groups.append((rings,mx.array(index),chirp,kernel))
        common = ['starts','costheta','seed','exponents','ca','cb']
        self.forward_kernel = mx.fast.metal_kernel(
            name='scalar_latitude_synthesis', input_names=common+['alm'],
            output_names=['out'], source=FORWARD, header=HEADER,
        )
        self.adjoint_kernel = mx.fast.metal_kernel(
            name='scalar_latitude_adjoint', input_names=common+['rings'],
            output_names=['out'], source=ADJOINT, header=HEADER,
        )
        self.alias_kernel = mx.fast.metal_kernel(
            name='scalar_ring_alias',
            input_names=['rings','phase','ring_of_pixel','nphi','ringstart'],
            output_names=['out'], source=ALIAS,
        )
        self.unalias_kernel = mx.fast.metal_kernel(
            name='scalar_ring_unalias',
            input_names=['fftmap','phase','nphi','ringstart'],
            output_names=['out'], source=UNALIAS,
        )
        mx.eval(self.seed, self.ca, self.cb, self.costheta, self.phase)

    def ring_fft(self, values, inverse):
        """Batched exact-length FFTs through power-of-two Bluestein convolutions."""
        rows = [None]*self.nr
        for rings, index, chirp, kernel in self.fft_groups:
            block = values[index]
            # conj(FFT(conj(x))) is the unnormalized inverse Fourier sum.
            # Synthesis needs that sum, not the inverse FFT divided by N.
            # Adjoint uses the ordinary negative-exponent forward sum.
            if inverse:
                block = mx.conj(block)
            if chirp is None:
                transformed = mx.fft.fft(block,axis=-1)
            else:
                transformed = mx.fft.ifft(
                    mx.fft.fft(block*chirp,axis=-1)*kernel,axis=-1,
                )*chirp
            if inverse:
                transformed = mx.conj(transformed)
            # Finish a group before exposing its row views to concatenation.
            # This also bounds the size of the pending execution graph.
            mx.eval(transformed)
            for row, ring in enumerate(rings):
                rows[ring] = transformed[row,:int(self.nphi[ring])]
        return mx.concatenate(rows)

    def synthesis(self, alm):
        """Packed m>=0 coefficients to a real HEALPix RING map."""
        # First contract degrees l at fixed (ring,m). The resulting array
        # has a real/imaginary pair for each ring and nonnegative order.
        inputs = [self.starts,self.costheta,self.seed,self.exponents,self.ca,self.cb]
        rings = self.forward_kernel(
            inputs=inputs+[alm.view(mx.float32)],
            template=[('NR',self.nr),('NM',self.nm)],
            grid=(self.nr*self.nm,1,1),threadgroup=(128,1,1),
            output_shapes=[(self.nr,self.nm,2)],output_dtypes=[mx.float32],
        )[0]
        mx.eval(rings)
        # Then collect orders into each ring's actual Fourier bins. Short
        # polar rings have aliases: retain both signs of every m, but count
        # zero once. Finally evaluate the longitude waves at pixel positions.
        bins = self.alias_kernel(
            inputs=[rings,self.phase,self.rgpu,self.ngpu,self.sgpu],
            template=[('NP',self.npix),('NM',self.nm)],
            grid=(self.npix,1,1),threadgroup=(128,1,1),
            output_shapes=[(self.npix,2)],output_dtypes=[mx.float32],
        )[0].view(mx.complex64).reshape(self.npix)
        mx.eval(bins)
        return self.ring_fft(bins, inverse=True).real

    def adjoint(self, values):
        """Unweighted pixel adjoint in the real-sky harmonic inner product."""
        # Reverse the forward sequence: ring Fourier sums, undo longitude
        # phases/aliases, then sum latitude weights back into coefficients.
        # This is a pixel sum without pixel-area weighting. The solve scales
        # both sides by pixel area; the raw adjoint contract stays explicit.
        fftmap = self.ring_fft(values.astype(mx.complex64), inverse=False)
        mx.eval(fftmap)
        rings = self.unalias_kernel(
            inputs=[fftmap.view(mx.float32),self.phase,self.ngpu,self.sgpu],
            template=[('NR',self.nr),('NM',self.nm)],
            grid=(self.nr*self.nm,1,1),threadgroup=(128,1,1),
            output_shapes=[(self.nr,self.nm,2)],output_dtypes=[mx.float32],
        )[0]
        mx.eval(rings)
        # Each 32-lane group sums a tile of latitude rings. A subsequent
        # device reduction combines tiles. No two groups race to update the
        # same coefficient, and no floating-point atomic addition is needed.
        partials = self.adjoint_kernel(
            inputs=[self.starts,self.costheta,self.seed,self.exponents,self.ca,self.cb,rings],
            template=[('NR',self.nr),('NM',self.nm),('NA',self.na)],
            grid=(self.tiles*self.nm*32,1,1),threadgroup=(32,1,1),
            output_shapes=[(self.tiles,self.na,2)],output_dtypes=[mx.float32],
        )[0]
        return mx.sum(partials,axis=0).view(mx.complex64).reshape(self.na)

    def solve(self, values, tolerance=1.e-6, maxiter=30):
        """Solve the unweighted least-squares map problem on the GPU.

        Conjugate gradients act on S^dagger S, with pixel area scaling both
        sides. This scalar scaling does not change the condition number. The target is the same unweighted least-squares
        problem as ducc0 pseudo_analysis, but the iteration is different.
        The final true normal residual is checked independently of the
        recursively updated residual. This does not claim a bound on the
        map's unrepresentable high-frequency power.
        """
        area = 4*np.pi/self.npix
        # Positive m represents both +m and -m in a real sky. Its norm
        # therefore has weight two; m=0 has weight one. Omitting this would
        # change the scalar products used to choose the CG step lengths.
        weight = mx.concatenate((mx.ones(self.nm),2*mx.ones(self.na-self.nm)))
        rhs = area*self.adjoint(values)
        solution = mx.zeros_like(rhs)
        residual = rhs
        direction = residual
        norm = mx.sum(weight*mx.abs(rhs)**2)
        square = norm
        mx.eval(norm)
        if float(norm.item()) == 0:
            return solution, {'iterations':0,'normal_residual':0.}
        # Matrix-free CG: action is H*p with H=area*S^dagger*S. No dense
        # normal matrix is stored. Each iteration needs a full forward and
        # adjoint pair, which is why moving only point arithmetic to GPU
        # gave little benefit in the earlier implementation.
        for iteration in range(1,maxiter+1):
            action = area*self.adjoint(self.synthesis(direction))
            denominator = mx.sum(weight*mx.real(mx.conj(direction)*action))
            # Move along the search direction, then subtract the implied
            # change in H*a from the residual b-H*a.
            alpha = square/denominator
            solution = solution+alpha*direction
            residual = residual-alpha*action
            next_square = mx.sum(weight*mx.abs(residual)**2)
            mx.eval(solution,residual,next_square)
            relative = float(mx.sqrt(next_square/norm).item())
            if not np.isfinite(relative):
                raise RuntimeError('GPU normal-equation iteration became nonfinite')
            if relative < tolerance:
                break
            direction = residual+(next_square/square)*direction
            square = next_square
        # Recursive residual updates can accumulate rounding error. Apply
        # H once more to the final solution and test the actual equation.
        # A small normal residual does not remove unresolved sky modes and
        # is not a substitute for comparison with the ducc0 reference.
        true_residual = rhs-area*self.adjoint(self.synthesis(solution))
        quality = mx.sqrt(mx.sum(weight*mx.abs(true_residual)**2)/norm)
        mx.eval(solution,quality)
        measured = float(quality.item())
        if not np.isfinite(measured) or measured > 3*tolerance:
            raise RuntimeError(f'GPU solve did not converge: normal residual {measured}')
        return solution, {'iterations':iteration,'normal_residual':measured}
