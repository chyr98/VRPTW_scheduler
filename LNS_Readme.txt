'--frac', '-f', type=float, default=0.25, help='The fraction of visits that will be removed and reinserted for each iteration of local search.')

'--benchmarks-dir', '-b', type=str, required=True,  help='The path of the benchmarks folder. Ex. if in same folder, then .\benchmarks'

'--prob-info', '-p', type=str, required=True, help='The details of the problem in the format of the benchmark names, i.e. _v1_c4_tw4_xy16_0')

'--nSwaps', '-s', type=str, default=2, help='The number of swaps'

'--optimal', '-o', type=str, default = 0, help='the optimal solution, if available. If not provided, valued at 0. '

'--optimality_gap', '-og', type=str, default = 0.0001, help = 'optimality gap, i.e 0.0001 is 0.01%.'

'--d', '-d', type=str, default = 5, help='parameter for limited discrepancy search'

'--t_lim', '-t', type=str, default = 6, help='total time limit in seconds'

'--t_lim_iter', '-ti', type=str, default = 5, help='iteration time limit in seconds'