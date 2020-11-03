import copy as cp
import argparse
import random

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

    all_visits = []



    return all_visits


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
        
    #get MPP input solution: the route of each vehicle is stored in a sublist in path. This is the input solution to LNS
    path = readPath(MPP_soln_file,nVeh)

    #get original paths/routes, same data form as the candidate routes
    original_path = readPath(original_soln_file,nVeh)

    #get a list of parameters for each visit in the routes [[customer number, vehichle served by, service time],...]
    all_visits = makeVisitData(path,cData)


    ###NOTE original data should use original service time windows
    #so need to add orig_cData




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

    args = parser.parse_args()

    if not 0 < args.frac < 1:
        print('frac must be between 0 and 1')

    else:
        print('running LNS...')
        soln = run_LNS(args.frac, args.benchmarks_dir, args.prob_info, args.nSwaps)

        print('saving solution...')
        write_Soln(soln)

