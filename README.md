# FairDrop

This is the companion code for the paper:
Spinelli I, Scardapane S, Hussain A, Uncini A, [FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning](https://ieeexplore.ieee.org/document/9645324).


### Fair edge dropout

We introduce a flexible biased edge dropout algorithm for enhancing fairness in graph representation learning. FairDrop targets the negative effect of the network's homophily w.r.t the sensitive attribute.

![Schematics of the proposed framework.](https://github.com/ispamm/FairDrop/blob/main/fairdrop.png)


### Acknowledgments

Many thanks to the authors of [[1]](https://github.com/aida-ugent/DeBayes) for making their code public and to the maintainers [[3]](https://github.com/rusty1s/pytorch_geometric) for such an awesome open-source library.



### Cite

Please cite [our paper](https://ieeexplore.ieee.org/document/9645324) if you use this code in your own work:

```
@ARTICLE{spinelli2021fairdrop,
  author={Spinelli, Indro and Scardapane, Simone and Hussain, Amir and Uncini, Aurelio},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TAI.2021.3133818}}
```
