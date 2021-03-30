import random as r
import matplotlib.pyplot as plt
from graph import Graph
import time
import os


def generate(n, filename, window):
    """
    Method used to generate a random graph of n nodes and write it on a file,
    the window indicates therange of values considered
    """
    file = open(filename, "w+")
    for i in range(n):
        coordinate1 = r.randint(0, window)
        coordinate2 = r.randint(0, window)
        string = str(coordinate1) + "  " + str(coordinate2) + "\n"
        file.write(string)


def readPath(filename):
    """
    Function used to read a permutation from a file
    :param filename:
    :return:
    """
    file = open(filename, "r")
    perm = []
    for f in file.readlines():
        perm.append(int(f) - 1)
    return perm

def GetCoordinates(filename):
    """
    Function used to get the coordinates of the nodes of a graph, stored in a textfile.
    """
    pairs = []
    file = open(filename, "r")
    for lines in file.readlines():
        position = lines.strip().split()
        pair = (int(float(position[0])), int(float(position[1])))
        pairs.append(pair)
    file.close()
    return pairs


def plotGraph(coordinates, permutation):
    """
    Function ysed to plot the points of a graph connected by a given tour (permutation)
    """
    x = [coordinates[i][0] for i in permutation]
    y = [coordinates[i][1] for i in permutation]
    x.append(coordinates[permutation[0]][0])
    y.append(coordinates[permutation[0]][1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y)
    plt.show()


def compareOptimalByVaringWindowSize():
    """
    Function used to test whether the window size varies the performance of Greedy,
    2-opt+Swap or the HGA.
    """
    n = 30
    values_genetic = []
    values_two_opt_swap = []
    values_greedy = []
    values = [1000 + 10 * x for x in range(50)]
    for i in values:
        generate(n, "citiesNew", i)
        two_opt_opt = Graph(-1, "citiesNew")
        two_opt_opt.swapHeuristic(12)
        two_opt_opt.TwoOptHeuristic(12)
        values_two_opt_swap.append(two_opt_opt.tourValue())
        greedy = Graph(-1, "citiesNew")
        greedy.Greedy()
        values_greedy.append(greedy.tourValue())
        genetic = Graph(-1, "citiesNew")
        genetic.genetic(PopulationSize=50, mutate_parameter=0.1, generationLimit=50, eliteSize=5, k=10,
                        percentage=0.2)
        values_genetic.append(genetic.tourValue())
    plt.xlabel("Window Size")
    plt.ylabel("Tour Value")
    plt.plot(values, values_greedy, label="Greedy")
    plt.plot(values, values_two_opt_swap, label="Two_opt + swap")
    plt.plot(values, values_genetic, label="HGA")
    plt.legend(loc="upper left")
    plt.show()


def compareEfficienceRandomCities():
    """
    Function used to calculate the tour value of different graphs with other algorithms
    and plots the result
    """
    values_greedy = []
    values_two_opt_swap = []
    values_HGA_max = []
    values_HGA_min = []
    values_HGA_average = []
    number_of_nodes = [2 + k for k in range(49)]

    for n in number_of_nodes:
        print(n)

        generate(n, "citiesNew", 1000)
        two_opt_opt = Graph(-1, "citiesNew")
        two_opt_opt.swapHeuristic(12)
        two_opt_opt.TwoOptHeuristic(12)
        values_two_opt_swap.append(two_opt_opt.tourValue())
        greedy = Graph(-1, "citiesNew")
        greedy.Greedy()
        values_greedy.append(greedy.tourValue())
        values_genetic = []
        for i in range(1):
            genetic = Graph(-1, "citiesNew")
            genetic.genetic(PopulationSize=50, mutate_parameter=0.1, generationLimit=50, eliteSize=5, k=10,
                            percentage=0.2)
            values_genetic.append(genetic.tourValue())
        values_HGA_max.append(max(values_genetic))
        values_HGA_min.append(min(values_genetic))
        values_HGA_average.append(sum(values_genetic) / len(values_genetic))
    plt.xlabel("Number of nodes")
    plt.ylabel("Tour Value")
    plt.plot(number_of_nodes, values_greedy, label="Greedy")
    plt.plot(number_of_nodes, values_two_opt_swap, label="Two_opt + swap")
    plt.plot(number_of_nodes, values_HGA_max, label="Maximum HGA")
    plt.legend(loc="upper left")
    plt.show()


def compareValueToOptimal():
    """
    Function used to calculate the tour value of different graphs with known result with 3 algorithms
    and plots the result comparing it with the optimal value
    """
    files = ["oliver30", "att48", "st70", "pr76", "kroD100", "lin105", "xqf131"]
    a = ["oliver30", "att48", "st70", "pr76"]
    values_greedy = []
    values_twoOpt_swap = []
    values_HGA = []
    for file in files:
        opt = Graph(-1, file)
        path = file + "path"
        opt.perm = readPath(path)
        opt_value = opt.tourValue()
        greedy = Graph(-1, file)
        greedy.Greedy()
        values_greedy.append(opt_value / greedy.tourValue())
        swap_two = Graph(-1, file)
        swap_two.swapHeuristic(swap_two.n)
        swap_two.TwoOptHeuristic(swap_two.n)
        values_twoOpt_swap.append(opt_value / swap_two.tourValue())
        HGA = Graph(-1, file)
        HGA.genetic(PopulationSize=50, mutate_parameter=0.1, generationLimit=50, eliteSize=5, k=10,
                    percentage=0.2)
        values_HGA.append(opt_value / HGA.tourValue())
    plt.xlabel("Graphs")
    plt.ylabel("Value")
    print(len(values_twoOpt_swap))
    print(len(values_greedy))
    print(len(values_HGA))
    plt.plot(files, values_greedy, label="Greedy")
    plt.plot(files, values_HGA, label="HGA")
    plt.plot(files, values_twoOpt_swap, label="2_Opt+Swap")
    plt.legend(loc="upper left")
    plt.show()


def compareTimeRandomCities():
    """
    Function used to calculate runtime of different algorithms when they calculate the best tour for d
    different graphs with different number of nodes. It plots the result
    """
    time_greedy = []
    time_HGA = []
    time_twoopt = []
    number_of_nodes = [2 + k for k in range(49)]

    for n in number_of_nodes:
        print(n)

        generate(n, "citiesNew", 1000)
        two_opt_opt = Graph(-1, "citiesNew")
        start = time.time()
        two_opt_opt.swapHeuristic(two_opt_opt.n)
        two_opt_opt.TwoOptHeuristic(two_opt_opt.n)
        end = time.time()
        time_twoopt.append(end - start)
        greedy = Graph(-1, "citiesNew")
        start = time.time()
        greedy.Greedy()
        end = time.time()
        time_greedy.append(end - start)
        genetic_times = []
        for i in range(1):
            genetic = Graph(-1, "citiesNew")
            start = time.time()
            genetic.genetic(PopulationSize=50, mutate_parameter=0.1, generationLimit=50, eliteSize=5, k=10,
                            percentage=0.2)
            end = time.time()
            genetic_times.append(end - start)

        time_HGA.append(sum(genetic_times) / len(genetic_times))
    plt.xlabel("Number of nodes")
    plt.ylabel("Time(s)")
    plt.plot(number_of_nodes, time_greedy, label="Greedy")
    plt.plot(number_of_nodes, time_twoopt, label="Two_opt + swap")
    plt.plot(number_of_nodes, time_HGA, label="Maximum HGA")
    plt.legend(loc="upper left")
    plt.show()


def generateGraph(filename):
    """
    Function used to generate the plot of a graph and the best tour obtained by HGA.
    """
    g = Graph(-1, filename)
    g.genetic(50, 0.1, 50, 5, 10, 0.2)
    coordinates = GetCoordinates(filename)
    plotGraph(coordinates, g.perm)


def setUp():
    """
    Function used to generate the files needed for testing
    """

    filenames = ["oliver30", "att48", "st70", "pr76", "kroD100", "lin105", "xqf131"]
    paths = ["oliver30path", "att48path", "st70path", "pr76path", "kroD100path", "lin105path", "xqf131path"]
    test = open("test_files/test", "r")
    fileslines = []
    file = []
    for line in test.readlines():
        line = line.strip()
        if line != "-1":
            file.append(line)
        else:
            fileslines.append(file[:])
            file = []

    test.close()
    for i in range(len(filenames)):
        tempFile = open(filenames[i], "w+")
        for j in range(len(fileslines[i])):
            line = fileslines[i][j]
            if j != len(fileslines[i]) - 1:
                tempFile.write(line)
                tempFile.write("\n")
            else:
                tempFile.write(line)
        tempFile.close()
    path = open("test_files/path","r")
    pathfiles = []
    pathing = []
    for p in path.readlines():
        line = p.strip()
        if line != "-1":
            pathing.append(line)
        else:
            pathfiles.append(pathing[:])
            pathing = []
    path.close()
    for i in range(len(paths)):
        tempFile = open(paths[i], "w+")
        for j in range(len(pathfiles[i])):
            line = pathfiles[i][j]
            if j != len(pathfiles[i]) - 1:
                tempFile.write(line)
                tempFile.write("\n")
            else:
                tempFile.write(line)
        tempFile.close()

    return fileslines
def plotGraphOfGenerations(filename):
    """
     Function used to plot the imporvement of the best tour over several generations using the HGA
    """
    g = Graph(-1,filename)
    values = g.genetic(PopulationSize=50, mutate_parameter=0.1, generationLimit=50, eliteSize=5, k=10,
                            percentage=0.2)
    x = [i for i in range(50 + 1)]
    plt.plot(x, values)
    plt.show()

def destructor():
    """
    Function used to delete all the files used in the testing process
    :return:
    """
    filenames = ["citiesNew","oliver30", "att48", "st70", "pr76", "kroD100", "lin105", "xqf131","oliver30path", "att48path", "st70path", "pr76path", "kroD100path", "lin105path", "xqf131path"]
    for file in filenames:
        os.remove(file)

def main():
    """
    compareEfficienceRandomCities()
    compareTimeRandomCities()
    """
    setUp()
    compareEfficienceRandomCities()
    # compareOptimalByVaringWindowSize()
    compareTimeRandomCities()
    # compareEfficienceRandomCities()
    # compareValueToOptimal()
    #plotGraphOfGenerations("oliver30")
    destructor()


main()
