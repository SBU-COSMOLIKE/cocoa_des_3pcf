# DES two-point and three-point likelihood

Cocoa links this directory into `cobaya/likelihoods/des_3pcf`. The example
YAML selects both theory and likelihood:

```yaml
theory:
  camb:
    extra_args:
      halofit_version: takahashi
      dark_energy_model: ppf
  bihalofit: {}
likelihood:
  des_3pcf.joint_shear:
    data_file: des_y3_joint1.dataset
```

The complete examples also specify cosmological parameters, nuisance
parameters and priors. CAMB is configured there; the likelihood never
starts a private CAMB calculation. BiHalofit reads CAMB products through
Cobaya and supplies `matter_bispectrum`. A different theory block can
supply that same product. The likelihood has no BiHalofit import.

For paired feedback, keep the absolute bispectrum producer and add
`bacco_bispectrum` in the theory section. Set `external_baryon_response: true`
in this likelihood. Its `baryon_response` product supplies both ratios:
$`P_H=R_P P_D`$ and $`B_H=R_B B_D`$. The three-point model retains DMO power
for quadratic IA; the two-point core receives the once-corrected nonlinear
power through its existing setter. Linear TATT power and the fixed covariance
stay unchanged. No BACCO module is imported by this likelihood. See
[the commented setup in example 1](../EXAMPLE_EVALUATE1.yaml).

`joint_shear.py` selects the probe. `survey.py` holds the survey readers:
the functions that read the measurement tables, the joint covariance and
the source distributions named by the `.dataset`, used by `initialize` and
by notebooks that need the raw rows. `_cosmolike_prototype_base.py` follows
LSST Y1: `initialize` reads the dataset and configures the native library;
`get_requirements` requests theory products; `set_cosmo_related` and
`set_source_related` supply cosmology and nuisance inputs;
`internal_get_datavector` assembles xi+/xi-/Map3; `compute_logp` calls the
native joint likelihood. Both probes use this project's compiled module.
The source reconstruction, finite-triangle cuts and within-bin xi
averaging remain explicit scientific choices in `joint_shear.yaml`, as are
the Map3 side bins of the measurement (`theta_min_arcmin`,
`theta_max_arcmin`, `ntheta`) and the scale cut `theta_lower_cut_arcmin`
that selects the side bins entering the Map3 rows.

The `.dataset` uses the usual `key = value` format. It names local source,
measurement and covariance files, plus an optional mask and compression.
Synthetic data are fixed text inputs, prepared at the stated fiducial;
they are not regenerated at each sampled point. All shipped examples use
CosmoLike's IA normalization 0.01389 and redshift pivot 0.62.

The C++ `IP3PCF` singleton holds data, selection, covariance and its
factorization. It retains the two-/three-point cross-covariance. Its
Gaussian density includes the normalization needed for evidence; Cobaya
adds priors. With `mock_count > 0`, it instead evaluates the normalized
Percival density. This assumes a raw sample covariance from independent
Gaussian mocks, independent data, and any compression trained independently
of that covariance. The supplied footprint mocks do not by themselves
establish all of those assumptions. Example 4 requires a separately
prepared pilot compression and is a methodological demonstration.

`compression.py` constructs optional fixed linear transformations for the
compression study. The production likelihood reads its saved matrix and
row labels directly; it does not construct a Python compression object. Native
initialization applies the same transformation to data and both covariance
axes; subsequent predictions receive it before scoring. Compression is
not required for a joint forecast.

Run from the activated Cocoa directory:

```bash
cobaya-run projects/des_3pcf/EXAMPLE_EVALUATE1.yaml
```

See the [project README](../README.md) for data provenance, Nautilus examples
and the remaining differences from the published analysis.
