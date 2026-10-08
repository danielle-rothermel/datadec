# DataDecide paper source

These files support the claim quote lookups in
[`scripts/repro/claims.py`](../../../scripts/repro/claims.py):

- `example_paper.tex` — main paper text and appendix.
- `tables/pred_error.tex` — scaling-law prediction-error table.

Both files are copied byte-for-byte from the [arXiv v2 source
archive](https://arxiv.org/src/2504.11393v2). Line numbers and character offsets
in the claim inventory refer to these exact files. This is the text subset
needed for quote lookup; figures, bibliography, and LaTeX support files needed
to compile the paper are not included.

## Attribution and license

*DataDecide: How to Predict Best Pretraining Data with Small Experiments*,
Ian Magnusson, Nguyen Tai, Ben Bogin, David Heineman, Jena D. Hwang,
Luca Soldaini, Akshita Bhagia, Jiacheng Liu, Dirk Groeneveld, Oyvind Tafjord,
Noah A. Smith, Pang Wei Koh, and Jesse Dodge (2025).

Version: [arXiv:2504.11393v2](https://arxiv.org/abs/2504.11393v2),
revised July 13, 2025. The paper is distributed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
as linked by its arXiv record. The bundled source files are unmodified.

## SHA-256 provenance

```text
20dc7aa3f920fe465ddf2e12d6f72fff6e8bb3567f53e34f5555a6da138542d1  2504.11393v2.tar.gz (upstream archive; not bundled)
14c1c878f55f7644b74eeec308e8f829075b2c28e62fb1eff9ad0901b61bb599  example_paper.tex
5510c35ceedc7accbe114bdb1cc7f1ae232ffc222eaa62ea55993e5b71c6712e  tables/pred_error.tex
```
