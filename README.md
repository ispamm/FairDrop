# FairDrop

This is the companion code for the paper:
Spinelli I, Scardapane S, Hussain A, Uncini A, [Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning](https://arxiv.org/abs/2104.14210).


### Fair edge dropout

We introduce a flexible biased edge dropout algorithm for enhancing fairness in graph representation learning. FairDrop targets the negative effect of the network's homophily w.r.t the sensitive attribute.

![Schematics of the proposed framework.](https://github.com/spindro/FairDrop/blob/master/fairdrop.pdf)


### Acknowledgments

Many thanks to the authors of [[1]](https://github.com/aida-ugent/DeBayes) for making their code public and to the maintainers [[3]](https://github.com/rusty1s/pytorch_geometric) for such an awesome open-source library.



### Cite

Please cite [our paper](https://arxiv.org/abs/2104.14210) if you use this code in your own work:

```
@misc{spinelli2021biased,
    title={Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning},
    author={Indro Spinelli and Simone Scardapane and Amir Hussain and Aurelio Uncini},
    year={2021},
    eprint={2104.14210},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```