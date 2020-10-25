# MPP in VRPTW

## VRPTW Generator

```bash
python3 problem_generator.py \
  --name problem1
  --num-vehicles 2 \
  --x-limit 10 \
  --y-limit 10 \
  --num-customers 5
  --tw-limit-ratio 0.3
  --output problem1.txt
  --solution solution_problem1.txt
  --seed 1234
```

- `--name/-n`: the name of problem (default: 'VRPTW instance')
- `--num-vehicles/-v`: the number of vehicles (default: 2)
- `--x-limit/-x`: the max value of the x axis (default: 10)
- `--y-limit/-y`: the max value of the y axis (defualt: 10)
- `--num-costomers/-c`: the number of customers including the depot (default :5)
- `--tw-limit-ratio/-t`: the larger this value is the broader the time windows are (defualt: 0.4)
- `--output/-o`: the filename to write the problem (required)
- `--solution/-s`: the filename to write a solution used for the problem generation (optional)

## MPP Generator
Generate MPP by swapping two customers in the original routes.

```bash
python3 mpp_generator.py \
  --name problem1
  --num-vehicles 2 \
  --x-limit 10 \
  --y-limit 10 \
  --num-customers 5
  --tw-limit-ratio 0.3
  --num-perturbations 3
  --original-output problem1.txt
  --perturbated-output perturbated_problem1.txt
  --solution solution_problem1.txt
  --seed 1234
```

- `--num-perturbations/-m`: the number of swaps
- `--original-output/-o`: the filename to write the original problem (required)
- `--perturbated-output/-p`: the filename to write the perturbated problem (required)
- `--solution/-s`: the filename to write an original solution used for the orgiginal problem generation (optional)
- `--pertrubated-solution/-q`: the filename to write a solution used for the perturbated problem generation (optional)

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
