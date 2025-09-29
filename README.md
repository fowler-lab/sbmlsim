[![Tests](https://github.com/fowler-lab/sbmlsim/actions/workflows/tests.yaml/badge.svg)](https://github.com/fowler-lab/sbmlsim/actions/workflows/tests.yaml)

# sbmlsim
Python package to create upsampled/artificial datasets of alleles of bacterial genes to test training advanced machine learning models such as graph-based convolutionary neural nets. If successful, then real data can be collected to train the model.

## high-level design
* class-based design with `Sample` or `Batch` objects e.g.

```
batch = sbmlsim.Batch(n_samples=10,...)

or

for i in range(n_samples):

    samples = sbmlsim.Sample(n_res=3, n_sus=2, resistant_mutatations = options.resistant_mutations, random_seed=42...)
```

## research outputs
To be added once pre-printed.
