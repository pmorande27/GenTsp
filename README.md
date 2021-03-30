# GenTsp
GenTsp is a Hybrid Genetic Algorithm used to obtain High Quality answers for TSP optimisation problem in  polynomial time.
## Introduction
The TSP is a well known NP-hard problem, meaning that there is no current algorithm able to give the overall best possible solution in polynomial time for this problem. Many techniques have been proposed to get approximate answers for this problem. GenTSP implements a Hybrid Genetic Algorithm that aims to provide high quality answers for this problem. This algorithm has proven to be more effective than other simplier heursitics such as 2-Opt ot Greedy.

## Description
Genetic algorithms (GA) are inspired in biological process of Natural selection. This process dominates the evolution of species in the biological world, the main idea behind it is that the individuals with the best genes have higher probabilities of surviving. Therefore, the best genes (Qualities) of each generation are more likely to pass to the next one. This concept can be used as an Heuristic method to get good solutions to optimisation problems such as TSP.
<br><br>

In Genetic Algorithms it is usual to start with a random population of feasible solutions, then that population is
ranked according to a fitness function that is problem dependant, the individuals with a higher fitness
are more likely to breed. The old population is replaced by the new one (which has the ’genes’ of the best parents),
To maintain diversity and to be sure that the algorithm does not get stuck in a local minimum so easily, each
offspring has a probability of mutation.
<br><br>
Genetic Algorithms are expected to converge into the optimal solution as the best results from one generation
are taken to the next one and there is still room for variability due to mutations (in case that the algorithm initially
converges into a local minimum). The hybrid part of the algorithm comes from the fact that usually traditional GA
take a high number of generations to converge into an optimal solution. Therefore, 2-opt heuristic was implemented
inside the genetic algorithm to make the algorithm converge faster
