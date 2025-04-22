# KAnonymization
Paper: Liang, Yuting, and Reza Samavi. "Optimization-based k-anonymity algorithms." Computers & Security 93 (2020): 101753.

This repo contains official code for the implementation of the algorithms in the paper ["Optimization-based anonymity algorithms"](https://www.sciencedirect.com/science/article/pii/S0167404820300377). 

# Cite
If you find the content useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{LIANG2020101753,
    title = {Optimization-based k-anonymity algorithms},
    journal = {Computers & Security},
    volume = {93},
    pages = {101753},
    year = {2020},
    issn = {0167-4048},
    doi = {https://doi.org/10.1016/j.cose.2020.101753},
    url = {https://www.sciencedirect.com/science/article/pii/S0167404820300377},
    author = {Yuting Liang and Reza Samavi},
    keywords = {Anonymization, Optimization, Privacy, Security, Mixed Integer Linear Program},
    abstract = {In this paper we present a formulation of k-anonymity as a mathematical optimization problem. In solving this formulated problem, k-anonymity is achieved while maximizing the utility of the resulting dataset. Our formulation has the advantage of incorporating different weights for attributes in order to achieve customized utility to suit different research purposes. The resulting formulation is a Mixed Integer Linear Program (MILP), which is NP-complete in general. Recognizing the complexity of the problem, we propose two practical algorithms which can provide near-optimal utility. Our experimental evaluation confirms that our algorithms are scalable when used for datasets containing large numbers of records.}
}
```
