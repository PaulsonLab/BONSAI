# BONSAI - Bayesian Optimization of Network Systems under Uncertainty
Most real-world systems consists of uncertainty and we often need to optimize the design variables of these systems in the presence of adversarial variables . Robust optimization is one such methodolgy which considers optimization in the presence of uncertainty. Oftentimes, we come accross optimization of systems with multi-disciliniary teams working on design of systems with various nodes sharing design and uncertainty variables. Below we illustrate one such example containing aircraft design problem




## Usage
This repository also includes a .yml file of the conda environment which was used during development of the code base. [bonsai_env.yml](https://github.com/PaulsonLab/BONSAI/tree/main/BONSAI/bonsai_env.yml)

We have included all nine case studies + the illustrative example that is described in the paper for the reader's reference and reproducibility of results. You can find it in [BONSAI/case_studies](https://github.com/PaulsonLab/BONSAI/tree/main/BONSAI/case_studies). All the scripts are important but we will decribe the scripts that the user can run to replicate our results
* [search_process_step_one.py](https://github.com/PaulsonLab/BONSAI/blob/main/BONSAI/search_process_step_one.py): This is used to search over the joint design-uncertainty space using different baselines including BONSAI, ARBO, logEI, and BOFN. The search is initiated with $2(n_x + n_w) + 1$ initial points and we keep a fixed search budget of 100
* [recommendation_step_two.py](https://github.com/PaulsonLab/BONSAI/blob/main/BONSAI/recommendation_step_two.py): This is used to recommend the searched locations based on the posterior GP obtained from observed data. We have four options for recommenders - GP, GP-Quantile, GPFN and GPFN-Quantile
* [wc_performance_step_three.py](https://github.com/PaulsonLab/BONSAI/blob/main/BONSAI/wc_performance_step_three.py): This is used to observe the worst-case performance of the recommended design from the recommendation step. The code also contains some initial versions of the plotting seen in the paper (different colors, different styles)

Rest of the .py files have description and comments for further dissection. Please feel free to reach out for additional usage questions!
