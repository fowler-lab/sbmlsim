# sbmlsim
Create upsampled/artificial datasets of bacterial alleles for training machine learning models

## design thoughts
* class-based design with `Sample` or `Batch` -- the latter with have the iterator within the object. Plan is to try `Batch`

```
batch = sbmlsim.Batch(n_samples=10,...)

or

for i in range(n_samples):

    samples = sbmlsim.Sample(n_res=3, n_sus=2, resistant_mutatations = options.resistant_mutations, random_seed=42...)
```

* methods for producing output

## coding thoughts

* python package or CLI with an entrypoint `import sbmlsim` or `sbmlsim --n_res 3 ..` or both -> package first
* protect `main`? -> not sure we can as is a private repo so don't push to `main`!
* use pull requests where we ask one other people for a review
* unit tests run by GitHub Actions on push -> simple system just for testing, used SARS-CoV-2, or HPV, or made up virus
* use `black` as makes easier to read, run as we go, maybe install an extension in VSCode to run `black` automatically upon save
* use linting?? No.
* Google convention for docstrings

## how to write?

* post in `#general` and then push when done
* 

