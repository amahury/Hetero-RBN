# Hetero-RBN
This repository contains the code that was used to run the simulations mentioned in the paper "Temporal, structural, and functional heterogeneities extend criticality and antifragility in random Boolean networks", whose preprint can be found at https://arxiv.org/abs/2209.07505

# Introductory Guide
In addition to the README.md file, there are three other Python scripts in this repository. The first and most important is mi_rbn.py, this file is a library where you can find the necessary to build a random Boolean network and evolve its dynamics. The ConfigurationModel function contained in mi_rbn.py allows the reader to explore other types of heterogeneity in a random Boolean network. To do so, it is possible to modify the node distribution (structural heterogeneity) and the activation probability distribution (functional heterogeneity). In addition to the above, this first script allows to visualize the dynamic transitions of the nodes in the network in the style of a one-dimensional cellular automaton.

The second script is complexity.py, whose function is to calculate the connectivity versus complexity curves for the different homogeneity/heterogeneity combinations described in the article. 

The third and last script is antifragility.py, whose function is to calculate the antifragility curves with respect to the number of perturbed nodes (X) and the frequency at which the perturbations are applied (O).
