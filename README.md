# MPP in VRPTW

## VRPTW Generator

```bash
python3 problem_generator.py \
  --name problem1
  --num-vehicles 2 \
  --x-limit 10 \
  --y-limit 10 \
  --num-customers 5
  --tw-limit 3
  --output problem1.txt
  --solution solution_problem1.txt
  --seed 1234
```

- `--name/-n`: the name of problem (default: 'VRPTW instance')
- `--num-vehicles/-v`: the number of vehicles (default: 2)
- `--x-limit/-x`: the maxumum value of the x axis (default: 10)
- `--y-limit/-y`: the maxumum value of the y axis (defualt: 10)
- `--num-costomers/-c`: the number of customers including the depot (default :5)
- `--tw-limit/-t`:  The half of the maximum width of time windows (defualt: 3)
- `--output/-o`: the path to save the generated problem (required)
- `--solution/-s`: The path to save the solution used for generating the problem (optional)

## MPP Generator
Generate MPP by swapping two customers in the original routes.
If `--original-problem` and `--original-solution` exist, they will be used.
Otherwise, they will be generated accodring to the same options as `problem_generator.py`.
Note that `--tw-limit` should be set appropriately even if an existing problem and solution are used.

```bash
python3 mpp_generator.py \
  --name problem1
  --num-vehicles 2 \
  --x-limit 10 \
  --y-limit 10 \
  --num-customers 5
  --tw-limit 3
  --num-perturbations 3
  --original-output problem1.txt
  --perturbated-output perturbated_problem1.txt
  --solution solution_problem1.txt
  --seed 1234
```

- `--tw-limit/-t`:  The half of the maximum width of time windows (defualt: 3)
- `--num-perturbations/-m`: the number of swaps (default: 3)
- `--original-problem/-o`: the path of the original problem (required)
- `--perturbated-problem/-p`: the path to save the perturbated problem (required)
- `--solution/-s`: The path of the original solution (required)
- `--pertrubated-solution/-q`: the path to save the solution used for generating the perturbated problem (optional)

### Problem Format

```
# problem1
2
5
1 6 0.0 0.0
6 6 5 8
9 9 15 19
9 3 5 12
8 9 11 16
```
- the first line is problem name
- the second line is the number of vehicles
- the third line is the number of customers
- the subsequenting lines describe customers
  - each line describes one customer (the first line is the depot)
  - the first column is the x-coordinate
  - the second column is the y-coordinate
  - the third column is the ready time
  - the fourth column is the due time

### Solution Format

```
# prolem1
2
1 2
3 4
```
- the first line is problem name
- the second line is the number of vehicles
- the subsequenting lines describe routes
  - each line describes one route
  - routes are listed in the order of the indices of the vehicles
  - customers are listed in the order of the visits in each line
  - the depot is omitted

## Solution Validator

```bash
python3 validate_solution.py --problem problem1.txt --solution solution_problem1.txt
vehicle 0
        0: 0.0, 3: 8.54, 4: 14.63, 2: 15.63
vehicle 1
        0: 0.0, 1: 5.00
total cost: 34.17
```

- output routes of vehicles and the starting time at each customer
