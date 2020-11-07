Instructions to run the LNS code:

Output:
For each run of each problem instance, such as '_v4_c16_tw16_xy16_0', a results file (ie'LNS_Results_v4_c16_tw16_xy16_0') is written in a directory created in the same folder as LNS.py.
If such a file already exists, it is overwritten.

Results file format:
- ***optional (depending if this feauture was used) - time at which optimal objective was found. 
- best objective found
- path of vehicle 0
- path of vehicle 1
...
- path of vehicle (m-1)


Input Instructions:
It is assumed that the problem instances to run are located in a directory, and have the same names as in benchmarks. It is assumed that the directory has four files for each instance instance, for example:
- solution_v1_c4_tw4_xy16_0
- original_v1_c4_tw4_xy16_0
- perturbated1_solution_v1_c4_tw4_xy16_0
- perturbated1_v1_c4_tw4_xy16_0

Basic Arguments
-b: is the path to the directory (i.e .\benchmarks if in same folder)
-p: problem data, in the form "_v1_c4_tw4_xy16_0"
-s: number of swaps for the perturbation instance, assumed to be 2 by default
-t: total runtime of the algorithm (120 s by default)

Extra Arguments:
These arguments can be included if an optimal solution is known, and we want the algorithm to stops when it finds it (and record the time). 
-o: The value of the known optimal solution
-og: Optimality gap, as a decimal. The algorithm will stop when the best solution it finds is less than ("og"*"o") away from the optimal solution.  





Full argument descriptions:


'--frac', '-f', type=float, default=0.25, help='The fraction of visits that will be removed and reinserted for each iteration of local search.')

'--benchmarks-dir', '-b', type=str, required=True,  help='The path of the benchmarks folder. Ex. if in same folder, then .\benchmarks'

'--prob-info', '-p', type=str, required=True, help='The details of the problem in the format of the benchmark names, i.e. _v1_c4_tw4_xy16_0')

'--nSwaps', '-s', type=str, default=2, help='The number of swaps'

'--optimal', '-o', type=str, default = 0, help='the optimal solution, if available. If not provided, valued at 0. '

'--optimality_gap', '-og', type=str, default = 0.0001, help = 'optimality gap, i.e 0.0001 is 0.01%.'

'--d', '-d', type=str, default = 5, help='parameter for limited discrepancy search'

'--t_lim', '-t', type=str, default = 6, help='total time limit in seconds'

'--t_lim_iter', '-ti', type=str, default = 5, help='iteration time limit in seconds'