# spurs



## Install

`pip install spurs`

`conda install spurs`

## How to use

Installation will create a command line script `spurs`:

```bash
spurs 20150608_20170808.int
```
By default, will output to file `20150608_20170808.unw` matching the name.

To increase the tolerance (from `pi/10` radians) for faster convergence, showing iteration stats:
```bash
spurs 20150608_20170808.int -o 20150608_20170808.unw --tol .5 --debug
```

See `spurs --help` for all options.

Note that for input interferograms which aren't complex, float32 binary format, `gdal` must be installed. E.g. for a VRT input:

```bash
spurs 20150608_20170808.vrt -o 20150608_20170808.unw
```



spurs is an open source implementation of [1]:

## References

1. Chartrand, Rick, Matthew T. Calef, and Michael S. Warren. "Exploiting Sparsity for Phase Unwrapping." IGARSS 2019-2019 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2019.
