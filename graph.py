import random as r
import math
import time
import tsplib95



def euclid(p, q):
    x = p[0] - q[0]
    y = p[1] - q[1]
    return int(math.sqrt(x * x + y * y)+0.5)


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
                pair = (int(float(position[0])), int(float(position[1])))
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
        elif n < -1:
            problem = tsplib95.load(filename)
            nodes = list(problem.get_nodes())
            self.n = len(nodes)
            pairs = []
            for i in range(self.n):
                coords = list(problem.node_coords[i+1])
                pairs.append((int(coords[0]),int(coords[1])))
            self.dist = [[0 for i in range(self.n)] for j in range(self.n)]
            for k in range(self.n):
                for l in range(self.n):
                    self.dist[k][l] = euclid(pairs[k], pairs[l])

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
        if (self.tourValue() < current_value):
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
        self.perm[i:j] = reversed(self.perm[i:j])
        if (self.tourValue() < current_value):
            return True
        else:
            self.perm[i:j] = reversed(self.perm[i:j])
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

    def twoopt(self, tour, k):
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for j in range(2, self.n - 1):
                for i in range(1, j):
                    if self.tryReverse2(tour, i, j):
                        better = True

    def tryReverse2(self, tour, i, j):
        current_value = self.fitness(tour)
        tour[i:j] = reversed(tour[i:j])
        if (self.fitness(tour) < current_value):
            return True
        else:
            tour[i:j] = reversed(tour[i:j])
            return False

    def create_sample(self):
        sample = []
        for i in range(self.n - 1):
            sample.append(i + 1)
        return sample

    def initialize_random_perm(self):
        sample = self.create_sample()
        perm = [0]
        route = r.sample(sample, len(sample))
        route = perm + route
        return route

    def create_population(self, population_size):
        population = []
        for i in range(population_size):
            population.append(self.initialize_random_perm())
        return population

    def fitness(self, tour):
        value = 0
        for i in range(self.n):
            value += self.dist[tour[i]][tour[(i + 1) % self.n]]
        return value

    def getTotalFitness(self, generation):
        value = 0
        for x in range(len(generation)):
            value += self.fitness(generation[x])
        return value

    def Greedy(self):
        new = self.perm[0]
        unused = self.perm[1:self.n][:]
        for i in range(1, self.n):
            closest = min([(self.dist[new][j], j) for j in unused], key=lambda t: t[0])[1]
            self.perm[i] = closest
            unused.remove(closest)
            new = closest

    def breed_2(self, parent_one, parent_two):
        child = [0]
        index_one = 1
        index_two = 1
        for i in range(self.n - 1):
            try:
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
                if (dist_two == 0 or dist_one == 0):
                    print("hola")
                weights = [1 / dist_one, 1 / dist_two]
                choice = r.choices([way_one, way_two], weights=weights, k=1)[0]
                if choice == way_one:
                    index_one += 1
                if choice == way_two:
                    index_two += 1
                child.append(choice)
            except IndexError:
                print("hola")

        return child

    def mutate_2(self, generation, mutate_parameter):

        mutatedPop = [self.mutate_individual(individual, mutate_parameter) for individual in generation]
        return mutatedPop

    def rankGeneration(self, generation):
        gen = [(tour, 1 / self.fitness(tour)) for tour in generation]
        ordered = sorted(gen, key=lambda data: 1 / data[1])
        tours_ordered, probabilites = zip(*ordered)
        return tours_ordered, probabilites

    def NewGeneration(self, currentGeneration, eliteSize, mutation_rate, k, modified):
        currentGeneration, probalities = self.rankGeneration(currentGeneration)
        newGeneration = []
        for i in range(eliteSize):
            newGeneration.append(currentGeneration[i])
        length = len(currentGeneration) - eliteSize
        children = []
        for j in range(length):
            parents = r.choices(currentGeneration, weights=probalities, k=2)
            child_one = self.breed_2(parents[0], parents[1])
            children.append(child_one)

        children = self.mutate_2(children, mutation_rate)
        choices = r.choices(children, k=modified)
        for i in range(len(choices)):
            self.twoopt(choices[i], k)

        newGeneration += children
        return newGeneration

    def breed(self, parent_one, parent_two):
        cut = r.randint(1, len(parent_one) - 1)
        start = parent_one[:cut]
        end = [item for item in parent_two if item not in start]
        start.extend(end)
        return start

    def genetic_2(self, PopulationSize, mutate_parameter, generationLimit, eliteSize, k, percentage):
        modified = int(percentage * (PopulationSize - eliteSize))
        update = 0
        population = self.create_population(PopulationSize)
        for i in range(generationLimit):
            population = self.NewGeneration(population, eliteSize, mutate_parameter, k, modified)
            best = self.tourValue()
            for individual in population:
                if best > self.fitness(individual):
                    self.perm = individual[:]
            update += 1
            print(update)

    def mutate_individual(self, individual, mutationRate):
        for swapped in range(1, len(individual)):
            if (r.random() <= mutationRate):
                swapWith = r.randint(1, len(individual) - 1)
                city1 = individual[swapped]
                city2 = individual[swapWith]
                individual[swapped] = city2
                individual[swapWith] = city1
        return individual

def readPath(filename):
    file = open(filename,"r")
    perm = []
    for f in file.readlines():
        perm.append(int(f)-1)
    return perm

def main():
    b = Graph(-1,"oliver30")
    print(b.n)
    b.swapHeuristic(1000)
    b.TwoOptHeuristic(1000)
    print(b.perm)
    print(b.tourValue())
    b.perm = [i for i in range(30)]

    a = Graph(-1, "oliver30")
    a.Greedy()
    print(a.tourValue())
    # print(a.tourValue())
    """
    print(a.tourValue())
    a.swapHeuristic(12)
    print(a.tourValue())
    a.TwoOptHeuristic(12)
    print(a.tourValue())
    pop = a.create_population(10)
    newGen = a.NewGeneration(pop,3)
    newNew = a.NewGeneration(newGen,3)
    """
    start = time.time()
    """for i in range(10):
        a.perm = identity
        a.genetic_2(600,0.1,100,60)
        values.append(a.tourValue())
    """
    identity = [i for i in range(a.n)]
    a.perm = identity
    a.genetic_2(100, 0.05, 100, 10, 10,0.3)
    end = time.time()
    print(end - start)
    print(a.perm)
    print(a.tourValue())
    l = [x+1 for x in a.perm]
    l = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,24,26,27,28,29,30,2]
    a.perm = [x-1 for x in l]
    print(a.tourValue())




main()
