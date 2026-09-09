## 11. Reproduction

* Everything in this notebook is self-contained; `slt_sorting_lib.py` in the repository is the same
  code.  `SLT_FAST=1 jupyter nbconvert --execute slt_sorting.ipynb` runs a reduced version on CPU in a
  few minutes; the full run takes well under an hour on a single small GPU.
* The committed outputs were produced on a rented GPU through Hugging Face Jobs with
  `python run_on_hf_jobs.py launch --notebook slt_sorting.ipynb --flavor t4-small`; the script uploads
  the notebook to a private dataset repo, runs it with `papermill`, and brings back the executed
  notebook, figures and `results/`.
