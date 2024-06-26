# To do or not to do? Cost-sensitive causal classification with individual treatment effect estimates </br><sub><sub>Wouter Verbeke, Diego Olaya,  Marie-Anne Guerry, Jente Van Belle [[2023]](https://doi.org/10.1016/j.ejor.2022.03.049)</sub></sub>
Individual treatment effect models allow optimizing decision-making by predicting the effect of a treatment on an outcome of interest for individual instances. These predictions allow selecting instances to treat in order to optimize the overall efficiency and net treatment effect. In this article, we extend upon the expected value framework and introduce a cost-sensitive causal classification boundary for selecting instances to treat based on predictions of individual treatment effects and for the case of a binary outcome. The boundary is a linear function of the estimated individual treatment effect, the positive outcome probability and the cost and benefit parameters of the problem setting. It allows causally classifying instances in the positive and negative treatment class in order to maximize the expected causal profit, which is introduced as the objective at hand in cost-sensitive causal classification. We present the expected causal profit ranker which ranks instances for maximizing the expected causal profit at each possible threshold that results from a constraint on the number of positive treatments and which differs from the conventional ranking approach based on the individual treatment effect. The proposed ranking approach is experimentally evaluated on synthetic and marketing campaign data sets. The results indicate that the presented ranking method outperforms the cost-insensitive ranking approach.

## Repository structure
This repository is organised as follows:
```bash
|- notebooks/
    |- main.py
|- src/
    |- data/
        |- data.py
        |- fun_synthetic_data.py
    |- methods/
        |- _init_.py
        |- causaul_models.py
        |- performance.py
        |- stratified.py
```

## Data usage
Using this repository, the main file can only run on the synthetic data. For more information on the experiments with the ‘Bank’, ‘Criteo’, and ‘Hillstrom’ datasets, please contact:
jente.vanbelle@kuleuven.be
or
wouter.verbeke@kuleuven.be

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Citing
Please cite our paper and/or code as follows:

```tex

@article{verbeke2023,
  title={To do or not to do? Cost-sensitive causal classification with individual treatment effect estimates},
  author={Verbeke, Wouter and Olaya, Diego and Guerry, Marie-Anne and Van Belle, Jente},
  journal={European Journal of Operational Research},
  volume={305},
  number={2},
  pages={838--852},
  year={2023},
  publisher={Elsevier}
}

```
