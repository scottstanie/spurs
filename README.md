# sparsewrap



## Install

`pip install sparsewrap`

`conda install sparsewrap`

## How to use

```bash
sparsewrap 20150608_20170808.int
```
By default, will output to file `20150608_20170808.unw` matching the name.


```bash
sparsewrap 20150608_20170808.int -o 20150608_20170808.unw --debug --tol .5
```

For a input interferograms which aren't complex float32 binary format, `gdal` must be installed:

```bash
sparsewrap 20150608_20170808.vrt -o 20150608_20170808.unw
```


## References

1. Chartrand, Rick, Matthew T. Calef, and Michael S. Warren. "Exploiting Sparsity for Phase Unwrapping." IGARSS 2019-2019 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2019.
