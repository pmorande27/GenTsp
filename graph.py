import random as r
import math
import time
import matplotlib.pyplot as plt


def euclid(p, q):
    x = p[0] - q[0]
    y = p[1] - q[1]
    return math.sqrt(x * x + y * y)


class Graph:

    # Complete as described in the specification, taking care of two cases:
    # the -1 case, where we read points in the Euclidean plane, and
    # the n>0 case, where we read a general graph in a different format.
    # self.perm, self.dists, self.n are the key variables to be set up.
    def __init__(self, n, filename):
        file = open(filename, "r")
        self.perm = []
        self.n = 0
        if n == -1:
            pairs = []

            for lines in file.readlines():
                self.n += 1
                position = lines.strip().split()
                pair = (int((position[0])), int((position[1])))
                pairs.append(pair)

            self.dist = [[0 for i in range(self.n)] for j in range(self.n)]
            for k in range(self.n):
                for l in range(self.n):
                    self.dist[k][l] = euclid(pairs[k], pairs[l])

        elif n > -1:
            self.n = n
            self.dist = [[0 for i in range(self.n)] for j in range(self.n)]
            for lines in file.readlines():
                edge = lines.strip().split(" ")
                self.dist[int(edge[0])][int(edge[1])] = int(edge[2])
                self.dist[int(edge[1])][int(edge[0])] = int(edge[2])

        self.perm = [i for i in range(self.n)]

        # Complete as described in the spec, to calculate the cost of the

    # current tour (as represented by self.perm).
    def tourValue(self):
        value = 0
        for i in range(self.n):
            value += self.dist[self.perm[i]][self.perm[(i + 1) % self.n]]
        return value

    # Attempt the swap of cities i and i+1 in self.perm and commit
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self, i):
        current_value = self.tourValue()
        value_one = self.perm[i]
        value_two = self.perm[(i + 1) % self.n]
        self.perm[i] = value_two
        self.perm[(i + 1) % self.n] = value_one
        if self.tourValue() < current_value:
            return True
        else:
            self.perm[i] = value_one
            self.perm[(i + 1) % self.n] = value_two
            return False
        pass

    # Consider the effect of reversiing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.              
    def tryReverse(self, i, j):
        current_value = self.tourValue()
        self.perm[i:(j+1)] = reversed(self.perm[i:(j+1)])
        if self.tourValue() < current_value:
            return True
        else:
            self.perm[i:(j+1)] = reversed(self.perm[i:(j+1)])
            return False

    def swapHeuristic(self, k):
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for i in range(self.n):
                if self.trySwap(i):
                    better = True

    def TwoOptHeuristic(self, k):
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for j in range(self.n - 1):
                for i in range(j):
                    if self.tryReverse(i, j):
                        better = True

    def Greedy(self):
        """
        Used to construct a tour following a Greedy principle,
        always choose as next node the one that is closer.
        """
        new = self.perm[0]
        unused = self.perm[1:self.n][:]
        for i in range(1, self.n):
            closest = min([(self.dist[new][j], j) for j in unused], key=lambda t: t[0])[1]
            self.perm[i] = closest
            unused.remove(closest)
            new = closest

    def twoopt(self, tour, k):
        """
        Repeated version of twoopt, which works in place for a given tour
        """
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for j in range(self.n - 1):
                for i in range(j):
                    if self.tryReverse2(tour, i, j):
                        better = True

    def tryReverse2(self, tour, i, j):
        """
        Repeated version of tryReverse, which works in place for a given tour
        """
        current_value = self.getTourValue(tour)
        tour[i:(j+1)] = reversed(tour[i:(j+1)])
        if self.getTourValue(tour) < current_value:
            return True
        else:
            tour[i:(j+1)] = reversed(tour[i:(j+1)])
            return False

    def create_sample(self):
        """
        Function used to create the indentity permutation
        """
        sample = []
        for i in range(self.n):
            sample.append(i)
        return sample

    def initialize_random_perm(self):
        """
        Function used to generate a random valid tour.
        """
        sample = self.create_sample()
        route = r.sample(sample, len(sample))
        return route

    def create_population(self, population_size):
        """
        Function used to generate an array of random valid tours.
        """
        population = []
        for i in range(population_size):
            population.append(self.initialize_random_perm())
        return population

    def getTourValue(self, tour):
        """
        Function used to get the value of a given tour
        :return:
        """
        value = 0
        for i in range(self.n):
            value += self.dist[tour[i]][tour[(i + 1) % self.n]]
        return value

    def fitness(self, tour):
        """
        Function used to get the fitness of a tour,
        which has to be inversely proportional (in this case 1/x^2) to the value of the tour"
        """

        value = 0
        for i in range(self.n):
            value += self.dist[tour[i]][tour[(i + 1) % self.n]]
        return 1 / self.getTourValue(tour) ** 2

    def Greedy_Crossover(self, parent_one, parent_two):
        """
        Function used to generate an offspring from two parents
        :return:
        """
        choice = r.choices([parent_one[0], parent_two[0]], k=1)[0]
        index_one = 0
        index_two = 0
        child = [choice]
        if choice == parent_one[0]:
            index_one = 1
        if choice == parent_two[0]:
            index_two = 1
        for i in range(self.n - 1):
            way_one = parent_one[index_one]
            way_two = parent_two[index_two]
            while (way_one in child):
                index_one += 1
                way_one = parent_one[index_one]
            while (way_two in child):
                index_two += 1
                way_two = parent_two[index_two]
            dist_one = self.dist[child[i]][way_one]
            dist_two = self.dist[child[i]][way_two]
            weights = [1 / dist_one, 1 / dist_two]
            choice = r.choices([way_one, way_two], weights=weights, k=1)[0]
            if choice == way_one:
                index_one += 1
            if choice == way_two:
                index_two += 1
            child.append(choice)
        return child

    def mutate_population(self, generation, mutate_parameter):
        """
        Function used to mutate population
        """

        mutatedPop = [self.mutate_individual(individual, mutate_parameter) for individual in generation]
        return mutatedPop

    def rankGeneration(self, generation):
        """
        Function used to rank a generation according to the fitness of each tour
        """
        gen = [(tour, self.fitness(tour)) for tour in generation]
        ordered = sorted(gen, key=lambda data: 1 / data[1])
        tours_ordered, probabilites = zip(*ordered)
        return tours_ordered, probabilites

    def Breed(self, currentGeneration, eliteSize, mutation_rate, k, modified):
        """
        Function used to generate the next generation
        """
        currentGeneration, probalities = self.rankGeneration(currentGeneration)
        newGeneration = []
        for i in range(eliteSize):
            newGeneration.append(currentGeneration[i])
        length = len(currentGeneration) - eliteSize
        children = []
        for j in range(length):
            parents = r.choices(currentGeneration, weights=probalities, k=2)
            child_one = self.Greedy_Crossover(parents[0], parents[1])
            children.append(child_one)

        children = self.mutate_population(children, mutation_rate)
        choices = r.choices(children, k=modified)
        for i in range(len(choices)):
            self.twoopt(choices[i], k)
        newGeneration += children
        return newGeneration

    def genetic(self, PopulationSize, mutate_parameter, generationLimit, eliteSize, k, percentage):
        """
        Function used to run the genetic algorithm all together.
        """
        best_values = []
        modified = int(percentage * (PopulationSize - eliteSize))
        update = 0
        population = self.create_population(PopulationSize)
        population = sorted(population, key=lambda t: self.getTourValue(t))
        best = self.getTourValue(population[0])
        self.perm = population[0][:]
        best_values.append(best)
        for i in range(generationLimit):
            population = self.Breed(population, eliteSize, mutate_parameter, k, modified)
            population = sorted(population, key=lambda t: self.getTourValue(t))
            new_best_route = population[0]
            new_best = self.getTourValue(population[0])
            best_values.append(new_best)
            if best > new_best:
                self.perm = new_best_route[:]
            update += 1
        return best_values

    def sortRoutes(self, population):
        """
        Function used to sort the routes in ascending order of tour value
        """
        return sorted(population, key=lambda t: self.getTourValue(t))

    def mutate_individual(self, individual, mutationRate):
        """
        Function used to mutate and individual of a population
        """
        for swapped in range(len(individual)):
            if r.random() <= mutationRate:
                swapWith = r.randint(0, len(individual) - 1)
                city1 = individual[swapped]
                city2 = individual[swapWith]
                individual[swapped] = city2
                individual[swapWith] = city1
        return individual