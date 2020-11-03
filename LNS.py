import copy as cp
import argparse
import random
import math

def readPath(filepath,nVeh):
    #reads a solution file and returns a list of sublists for each vehicle route

    f = open(filepath,'r')
    f.readline()
    f.readline()

    #record the route of each vehicle in a sublist
    #this is the first candidate solution
    path = []
    for i in range(nVeh):
        params = [int(i) for i in f.readline().split()]
        path.append(params)

    return path

def makeVisitData(path, cData):
    #return a list of parameters for each visit in the routes [[customer number, vehichle served by, service time],...]

    nVeh = len(path)
    nCus = len(cData)

    

    all_visits = []

    for i in range(nVeh):
        #for each vehicle, add the depot as the starting node. No need to worry about returning to the depot - this has no impact on the MPP objective
        path[i] = [0] + path[i]
        t = 0

        for j in range((len(path[i])-1)):
            node_curr = path[i][j]
            node_next = path[i][j+1]


            #distances
            coords_curr = (cData[node_curr][0], cData[node_curr][1])
            coords_next = (cData[node_next][0], cData[node_next][1])

            d = distBtwnPts(coords_curr, coords_next)

            #get service time
            t = t + d


            #service time is delayed if need to wait for time windows
            tw_start = cData[node_next][2]
            tw_end = cData[node_next][3]

            if tw_start > t:
                t = tw_start
            elif tw_end < t:
                print("infeasible: (vehicle, node i, node j)" , (i, node_curr, node_next))

            info_next = [node_next, i, t]
            all_visits.append(info_next)
            

    return all_visits

def distBtwnPts(coords1,coords2):
    
    a = ((coords1[0]-coords2[0])**2 + (coords1[1]-coords2[1])**2)
    return (math.sqrt(a))
            
def removeVisits(path, all_visits, frac):
    #find visits to remove
    
    #number of visits to remove
    nRem = math.ceil(frac*len(all_visits))
   
    rem_vs = []
    red_path = []

    print(all_visits)

    #randomly choose first removal
    rand_index = random.randrange(len(all_visits))
    rem_vs.append(all_visits[rand_index])
    del all_visits[rand_index]

    while len(rem_vs) < nRem:

        #chose a random visit from the already removed set
        rand_index = random.randrange(len(rem_vs))
        v_compare = rem_vs[rand_index]

        #v_ind = removeVisit(v_compare,all_visits)



        break

    #delete all removed visits from the path
    for i in range(len(rem_vs)):
        j = rem_vs[i][0]
        vehicle = rem_vs[i][1]
        
        #don't iterate over path while modifiying

   
    return rem_vs, path


def removeVisit(v_comp, all_visits):
    #ranks all_visits based on similarity to v_comp, selects one visit to remove, removes from all_visits, appends to rem_vs

    pass
    #return v_ind


def run_LNS(frac, bPath, pInfo, nSwaps):
    #run the LNS algorithm
    #parameters: frac -  The fraction of visits that will be removed and reinserted for each iteration of local search.
    #bPath: benchmarks folder path
    #pInfo: details of the problem - will be added to create filepaths such as: "perturbated1_v4_c64_tw4_xy16_1.txt" 
    #nSwaps: number of swaps that the MPP was generated with

 

    #filepaths
    MPP_file = bPath + '\\perturbated' + str(nSwaps) + pInfo + '.txt'
    MPP_soln_file = bPath + '\\perturbated' + str(nSwaps) + "_solution" + pInfo + '.txt'
    original_soln_file = bPath + '\\solution' + str(pInfo) + '.txt'
    original_file = bPath +'\\original' + str(pInfo) + '.txt'

    #Read in MPP problem file
    f1 = open(MPP_file,'r')
    f1.readline()

    #get number of customers and vehicles
    nVeh = int(f1.readline())
    nCus = int(f1.readline())

    #read customer data, each sublist corresponds to one customer, with index matching customer index, and has data [xCoord, yCoord, twStart, twEnd]
    cData = []

    for i in range(nCus):
        params = f1.readline().split()
        int_params = [int(i) for i in params]
        cData.append(int_params)
        

    #read customer data from original problem
    f2 = open(original_file)
    
    #ignore first lines
    f2.readline()
    f2.readline()
    f2.readline()

    orig_cData = []

    for i in range(nCus):
        params = f2.readline().split()
        int_params = [int(i) for i in params]
        orig_cData.append(int_params)


    #get MPP input solution: the route of each vehicle is stored in a sublist in path. This is the input solution to LNS
    path = readPath(MPP_soln_file,nVeh)

    #get original paths/routes, same data form as the candidate routes
    original_path = readPath(original_soln_file,nVeh)

    #get a list of parameters for each visit in the routes [[customer number, vehichle served by, service time],...]
    all_visits = makeVisitData(cp.deepcopy(path),cp.deepcopy(cData))
    original_visits = makeVisitData(cp.deepcopy(original_path), cp.deepcopy(orig_cData))


    #get removed visits and the reduced path
    rem_vs, red_path = removeVisits(cp.deepcopy(path),cp.deepcopy(all_visits),float(frac))

   

    #TO DO: check copy for all list args

    soln = []
    return soln

def write_Soln(soln):

    pass
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac', '-f', type=float, default=0.25,
                        help='The fraction of visits that will be removed and reinserted for each iteration of local search.')
    parser.add_argument('--benchmarks-dir', '-b', type=str, required=True,
                        help='The path of the benchmarks folder')
    parser.add_argument('--prob-info', '-p', type=str, required=True,
                        help='The details of the problem in the format of the benchmark names, i.e. _v1_c4_tw4_xy16_0')
    parser.add_argument('--nSwaps', '-s', type=str, default=2, help='The number of swaps')
    parser.add_argument('--optimal', '-o', type=str, default = 0,
                        help='the optimal solution, if available. If not provided, valued at 0. ')

    args = parser.parse_args()

    if not 0 < args.frac < 1:
        print('frac must be between 0 and 1')

    else:
        print('running LNS...')
        soln = run_LNS(args.frac, args.benchmarks_dir, args.prob_info, args.nSwaps)

        print('saving solution...')
        write_Soln(soln)

