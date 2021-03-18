import random as r
def generate(n,filename):
    file = open(filename,"w")
    for i in range(n):
        coordinate1= r.randint(0,1000000)
        coordinate2 = r.randint(0,1000000)
        string = str(coordinate1)+ "  " + str(coordinate2)+"\n"
        file.write(string)
generate(100,"citiesNew")

