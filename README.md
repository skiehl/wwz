<a href="https://ascl.net/2310.003"><img src="https://img.shields.io/badge/ascl-2310.003-blue.svg?colorB=262255" alt="ascl:2310.003" /></a>

# wwz
A python3 weighted wavelet z-transform (WWZ) analysis package.

## Requirements

This script uses the following standard python packages:
+ datetime
+ pickle
+ sys

This script uses the following python packages:
+ numpy
+ scipy

The plotting script additionally uses:
+ matplotlib
+ os

## Getting Started

Get the python script:

    $ git clone https://github.com/skiehl/wwz.git

Open the jupyter notebook `Documentation.ipynb` for a demonstation of the
code. Either use jupyter lab:

    $ jupyter lab

Or jupyter notebook:

    $ jupyter notebook Documentation.ipynb

## Usage

Usage of the package is demonstrated in the jupyter notebook
`Documentation.ipynb`.
A complete code documentation is given in `html/`.

## Citation

Bibtex:

```
@MISC{2023ascl.soft10003K,
       author = {{Kiehlmann}, Sebastian and {Max-Moerbeck}, Walter and {King}, Oliver},
        title = "{wwz: Weighted wavelet z-transform code}",
     keywords = {Software},
 howpublished = {Astrophysics Source Code Library, record ascl:2310.003},
         year = 2023,
        month = oct,
          eid = {ascl:2310.003},
        pages = {ascl:2310.003},
archivePrefix = {ascl},
       eprint = {2310.003},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023ascl.soft10003K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


This software package is indexed on [ASCL](https://ascl.net/2310.003) and [ADS](https://ui.adsabs.harvard.edu/abs/2023ascl.soft10003K/).

## License

wwz is licensed under the BSD 3-Clause License - see the
[LICENSE](https://github.com/skiehl/wwz/blob/main/LICENSE) file.

## References

The WWZ method was introduced by
[Foster, 1996](https://ui.adsabs.harvard.edu/abs/1996AJ....112.1709F/abstract).

## Alternatives

Some other python implementations of the WWZ method are:

+ [libwwz](https://pypi.org/project/libwwz/)
+ [eaydin/WWZ](https://github.com/eaydin/WWZ)
