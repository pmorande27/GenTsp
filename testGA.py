import math
import random
import matplotlib.pyplot as plt
import time

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

        if (n == -1):

            table = []
            points = []

            with open(filename, "r") as nodes:
                node_lines = nodes.readlines()
                n_nodes = len(node_lines)

                for node in node_lines:
                    point = [float(val) for val in node.split()]
                    points.append(point)

                for i in range(n_nodes):
                    dist_from_i = []
                    for j in range(n_nodes):
                        dist_from_i.append(euclid(points[i], points[j]))
                    table.append(dist_from_i)

            self.n = n_nodes
            self.dists = table

        elif (n >= 0):

            table = [[0 for i in range(n)] for _ in range(n)]
            points = []

            with open(filename, "r") as edges:
                edge_lines = edges.readlines()

                for edge in edge_lines:
                    points.append([int(val) for val in edge.split()])

                for point in points:
                    i = point[0]
                    j = point[1]

                    table[i][j] = point[2]
                    table[j][i] = point[2]

            self.n = n
            self.dists = table

        self.perm = [i for i in range(self.n)]

    # Complete as described in the spec, to calculate the cost of the
    # current tour (as represented by self.perm).
    def tourValue(self):
        value = 0

        for i in range(self.n):
            value += self.dists[self.perm[i]][self.perm[(i + 1) % self.n]]

        return value

    # Attempt the swap of cities i and i+1 in self.perm and
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self, i):
        bestRoute = self.tourValue()

        i_value = self.perm[i]
        next_i_value = self.perm[(i + 1) % self.n]
        self.perm[i] = next_i_value
        self.perm[(i + 1) % self.n] = i_value

        if bestRoute <= self.tourValue():
            self.perm[i] = i_value
            self.perm[(i + 1) % self.n] = next_i_value

            return False

        else:
            return True

    # Consider the effect of reversing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.

    def tryReverse(self, i, j):
        bestRoute = self.tourValue()

        self.perm[i: (j + 1)] = self.perm[i: (j + 1)][::-1]

        if bestRoute <= self.tourValue():
            self.perm[i: (j + 1)] = self.perm[i: (j + 1)][::-1]

            return False

        else:
            return True

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

    # Implement the Greedy heuristic which builds a tour starting
    # from node 0, taking the closest (unused) node as 'next'
    # each time.
    def Greedy(self):
        temp_perm = [0]
        index_min = 0

        while len(temp_perm) != self.n:
            i = temp_perm[-1]
            for j in range(1, self.n):
                j_not_visited = j not in temp_perm
                lower_distance = (self.dists[i][index_min] > self.dists[i][j]) or (self.dists[i][index_min] == 0)
                if j_not_visited and lower_distance:
                    index_min = j

            temp_perm.append(index_min)

        self.perm = temp_perm

    # -------------------------------- GENETIC ALGORITHM ---------------------------------------

    def route_fitness(self, route):
        """
        A measure of how good certain route is.
        The larger the route fitness, the less cost it has.
        """
        value = 0
        if len(route)!= self.n:
            print(route)
        assert len(route) == self.n

        for i in range(self.n):
            value += self.dists[route[i]][route[(i + 1) % self.n]]

        return 1 / value

    def randomise_route(self, route):
        """
        Given a route (given as a list of length self.n and starting at 0),
        permutes the elements in the route to return a new randomised route.
        """

        if route[0] != 0:
            print(f"First element isn't 0 in {route}")

        if len(route) != self.n:
            print(f"{route} doesn't have length {self.n}")

        random_route = random.sample(route[1:], self.n - 1)
        random_route.insert(0, 0)

        return random_route

    def rank_routes(self, routes):
        """
        Given a list of possible routes, ranks them according to fitness.
        :return: a list of tuples, with tuple[0] = index of route in routes and tuple[1] = fitness of route
                 a list of probabilities of propagation for each route;
                 a larger probability gives better chances of being promoted for the next generation
        """
        route_rankings = {}
        total_fitness = 0

        for i in range(len(routes)):
            fitness = self.route_fitness(routes[i])
            route_rankings[i] = fitness
            total_fitness += fitness

        probabilities = [self.route_fitness(routes[i]) / total_fitness for i in range(len(routes))]

        return sorted(route_rankings.items(), key=lambda x: x[1], reverse=True), probabilities

    def breed(self, routes,ranked_routes, probabilities, elite_size):
        """
        Selects routes to be mutated for the next generation.
        Combines elites (obtain best fitness value) with randomly selected routes,
        using the "Roulette Wheel Selection" method
        :param ranked_routes: an order list of tuples, according to fitness of a route
        :param probabilities: a list containing probabilities of success to next generation for each route
        :param elite_size: number of elements guaranteed to go on to the next generation
        :return: a list of indices, corresponding to the routes used for breeding the next generation
        """
        length = len(ranked_routes)-elite_size
        indices = [ranked_routes[i][0] for i in range(elite_size)]
        children = [routes[i] for i in indices]
        indices = [indices[0] for indices in ranked_routes]
        for i in range(length):

            parents_index = random.choices(indices,weights=probabilities,k= 2)
            parents = [routes[i] for i in parents_index]
            child = self.crossover(parents[0],parents[1])
            children.append(child)
        return children

    def select_from_routes_2(self, ranked_routes, probabilities, elite_size):
        """
        Selects routes to be mutated for the next generation.
        Combines elites (obtain best fitness value) with randomly selected routes,
        using the "Roulette Wheel Selection" method
        :param ranked_routes: an order list of tuples, according to fitness of a route
        :param probabilities: a list containing probabilities of success to next generation for each route
        :param elite_size: number of elements guaranteed to go on to the next generation
        :return: a list of indices, corresponding to the routes used for breeding the next generation
        """

        selected = [ranked_routes[i][0] for i in range(elite_size)]

        for i in range(len(ranked_routes) - elite_size):
            cutoff = random.random()
            for route in ranked_routes:
                index = route[0]
                if probabilities[index] <= cutoff:
                    selected.append(index)
                    break

        return selected

    def get_parents(self, routes, selected_route_indices):
        """
        Gets routes according to the indices that have been selected for breeding
        """
        return [routes[i] for i in selected_route_indices]

    def crossover(self, parent_a, parent_b):
        """
        Performs crossover for 2 parents; that is, randomly selects a slice of a parent,
        and fills it up with the route of the second parent, as to form a child route.
        """

        assert len(parent_a) == len(parent_b)


        cut_index = random.randint(1, len(parent_a) - 1)
        start = parent_a[:cut_index]
        end = [i for i in parent_b if i not in start]
        start.extend(end)

        if len(start) != self.n:
            print(f"The crossover gave: {start} with cut_index {cut_index}")

        return start
        #if (child_fitness >= parent_a_fitness and child_fitness >= parent_b_fitness):
        #    return start
        #else:
        #    if (parent_a_fitness > parent_b_fitness):
        #        return parent_a
        #    else:
        #        return parent_b

    def crossover_2(self, parent_a, parent_b):
        """
        Performs crossover for 2 parents; that is, randomly selects a slice of a parent,
        and fills it up with the route of the second parent, as to form a child route.
        """

        assert len(parent_a) == len(parent_b)

        cut_index_a = random.randint(1, len(parent_a) - 1)
        cut_index_b = random.randint(cut_index_a, len(parent_a) - 1)
        start = parent_a[cut_index_a: cut_index_b]
        start.insert(0, 0)
        end = [i for i in parent_b if i not in start]
        start.extend(end)

        if len(start) != self.n:
            print(f"The crossover gave: {start} with cut_index {cut_index_a}")

        return start

    def crossover_3(self, parent_a, parent_b):
        """
        Performs crossover for 2 parents; that is, randomly selects a slice of a parent,
        and fills it up with the route of the second parent, as to form a child route.
        """

        assert len(parent_a) == len(parent_b)

        cut_index_a = random.randint(1, len(parent_a) - 1)
        cut_index_b = random.randint(cut_index_a, len(parent_a) - 1)

        child_a = parent_a[cut_index_a: cut_index_b]
        child_a.insert(0, 0)
        child_a_end = [i for i in parent_b if i not in child_a]
        child_a.extend(child_a_end)

        child_b = parent_b[cut_index_a: cut_index_b]
        child_b.insert(0, 0)
        child_b_end = [i for i in parent_a if i not in child_b]
        child_b.extend(child_b_end)

        return [child_a, child_b]

    def crossover_routes(self, selected_routes, elite_size):
        """
        Performs crossover an all routes that were selected for breeding.
        Those elites are passed without performing crossover.
        """
        children = [selected_routes[i] for i in range(elite_size)]
        randomised_routes = random.sample(selected_routes, len(selected_routes))
        n_left = len(selected_routes) - elite_size

        for i in range(n_left):
            children.append(self.crossover(randomised_routes[i], randomised_routes[(i + 1) % n_left]))

        return children

    def crossover_routes_3(self, selected_routes, elite_size):
        """
        Performs crossover an all routes that were selected for breeding.
        Those elites are passed without performing crossover.
        """
        children = [selected_routes[i] for i in range(elite_size)]
        randomised_routes = random.sample(selected_routes, len(selected_routes))
        n_left = len(selected_routes) - elite_size

        i = 0

        while (len(children) <= len(selected_routes)):
            children.extend(self.crossover_3(randomised_routes[i], randomised_routes[(i + 1) % n_left]))
            i += 1

        return children

    def mutate(self, route, rate):
        """
        Randomly mutates a route, by swapping adjacent nodes.
        """
        for i in range(1, len(route)):
            if (random.random() <= rate):
                swap_index = random.randint(1, len(route) - 1)
                initial = route[i]
                route[i] = route[swap_index]
                route[swap_index] = initial

        return route

    def mutate_routes(self, routes, rate):
        """
        Mutate all routes
        """
        mutated = [self.mutate(route, rate) for route in routes]

        return mutated

    def next_gen(self, routes, elite_size, rate):
        ranked_routes, probabilities = self.rank_routes(routes)
        children = self.breed(routes,ranked_routes, probabilities, elite_size)
        return self.mutate_routes(children, rate)


    def genetic(self, size, elite_size, rate, generations):
        routes = [self.perm[:] for _ in range(size)]
        best_route = self.perm
        best_fitness = self.route_fitness(best_route)
        update = 0

        for i in range(generations):
            update += 1
            print(update)
            routes = self.next_gen(routes, elite_size, rate)
            for route in routes:
                fitness = self.route_fitness(route)
                if fitness > best_fitness:
                    best_route = route[:]
                    best_fitness = fitness

        self.perm = best_route

    def plot(self, size, elite_size, rate, generations):
        route_plot = []
        routes = [g.randomise_route(self.perm) for _ in range(size)]
        best_route = self.perm
        best_fitness = self.route_fitness(best_route)

        for i in range(generations):
            routes = self.next_gen(routes, elite_size, rate)
            for route in routes:
                fitness = self.route_fitness(route)
                route_plot.append(fitness)
                if fitness > best_fitness:
                    best_route = route
                    print(f"The best route is: {best_route}")
                    print(f"The best route has length: {1/fitness}")
                    best_fitness = fitness


        print("--------------------------------------\n")
        print(f"The best route is: {best_route}")
        print(f"The best route has length: {1 / best_fitness}")
        fig, ax = plt.subplots()
        ax.plot(route_plot, c="b")
        plt.show()
        return best_route


g = Graph(-1, "bays29")
start = time.time()
g.genetic(500,50,0.1,1000)
print(g.tourValue())
end = time.time()
print(end-start)


#g.genetic()
#print(g.perm)
#print(g.tourValue())