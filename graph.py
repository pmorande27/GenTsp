import math
import random

def euclid(p,q):
    x = p[0]-q[0]
    y = p[1]-q[1]
    return math.sqrt(x*x+y*y)
                
class Graph:

    # Complete as described in the specification, taking care of two cases:
    # the -1 case, where we read points in the Euclidean plane, and
    # the n>0 case, where we read a general graph in a different format.
    # self.perm, self.dists, self.n are the key variables to be set up.
    def __init__(self,n,filename):
        file = open(filename,"r")
        self.perm = []
        if n == -1:
            pairs = []
            self.n = 0
           
            for lines in file.readlines():
                self.n+=1
                position = lines.strip().split("  ")
                pair = (float(position[0]),float(position[1]))
                pairs.append(pair)
                
            self.dist = [[0 for i in range(self.n)]for j in range(self.n)]
            for k in range(self.n):
                for l in range(self.n):
                    self.dist[k][l] = euclid(pairs[k],pairs[l])
    
        elif n>-1:
            self.n = n
            self.dist = [[0 for i in range(self.n)]for j in range(self.n)]
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
            value += self.dist[self.perm[i]][self.perm[(i+1)%self.n]]
        return value

    # Attempt the swap of cities i and i+1 in self.perm and commit
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self,i):
        current_value = self.tourValue()
        value_one = self.perm[i]
        value_two = self.perm[(i+1)%self.n]
        self.perm[i] = value_two
        self.perm[(i+1)%self.n] = value_one
        if (self.tourValue()<current_value):
            return True
        else:
            self.perm[i] = value_one
            self.perm[(i+1)%self.n] = value_two
            return False
        pass


    # Consider the effect of reversiing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.              
    def tryReverse(self,i,j):
        current_value = self.tourValue()
        self.perm[i:j] = reversed(self.perm[i:j])
        if(self.tourValue()<current_value):
            return True
        else:
            self.perm[i:j] = reversed(self.perm[i:j])
            return False

    def swapHeuristic(self,k):
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for i in range(self.n):
                if self.trySwap(i):
                    better = True

    def TwoOptHeuristic(self,k):
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for j in range(self.n-1):
                for i in range(j):
                    if self.tryReverse(i,j):
                        better = True

                        
    # Implement the Greedy heuristic which builds a tour starting
    # from node 0, taking the closest (unused) node as 'next'
    # each time.
    def Greedy(self):
        new = self.perm[0]
        unused = self.perm[1:self.n][:]
        for i in range(1,self.n):
            closest  = min([(self.dist[new][j],j) for j in unused],key = lambda t: t[0])[1]
            self.perm[i] = closest
            unused.remove(closest)
            new = closest
def main():
    a = Graph(12,"twelvenodes")
    """
    print(a.tourValue())
     a.swapHeuristic(12)
     print(a.tourValue())
     a.TwoOptHeuristic(12)
     print(a.tourValue())
    """
    a.Greedy()
    print(a.tourValue())


main()