# RWRReg

This repository contains the reference code for the paper "Are Graph Convolutional Networks Fully Exploiting Graph Structure?".

In particular, it shows how it's very easy to implement the RWRReg regularization on existing GNN models for node classification.
The starting GraphSage model was taken from <https://github.com/williamleif/graphsage-simple/>.

If you use the code in this repository please cite the following paper.
```
@misc{buffelli2020graph,
      title={Are Graph Convolutional Networks Fully Exploiting Graph Structure?},
      author={Buffelli, Davide and Vandin, Fabio},
      year={2020},
      eprint={2006.03814},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Link to paper: <https://arxiv.org/abs/2006.03814>

## Requirements
* PyTorch>=1.2
* NetworkX 1.11 (version number is important!)
* SciPy

## Instructions
The first time you launch this program using random walk features, it will generate and
save random walk statistics for the considered dataset. 
This may take a while, specially for large graphs, and it will take space on disk.
This only happens the first time for each dataset, then the program will read the saved 
random walk statistics from disk.

* Train and validate GraphSage

```python model.py --num-experiments 100```

* Train and validate GraphSage-AD 

```python model.py --num-experiments 100 --feat-addition AD```

* Train and validate GraphSage-RW

```python model.py --num-experiments 100 --feat-addition RW```

* Train and validate GraphSage-RWReg

```python model.py --num-experiments 100 --feat-addition RW --rwr-reg --rwrreg-without-feat-addition```

* Train and validate GraphSage-RW+RWReg

```python model.py --num_experiments 100 --feat_addition RW --rwr_reg```

## License
Refer to the file [LICENSE](LICENSE).
