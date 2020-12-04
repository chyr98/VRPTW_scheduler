import argparse
import json


def average_result(result, methods, group_by_params, fields):
    summed_result = {}
    key_count = {}

    for r in result:
        method = r['methods']

        if method not in methods:
            continue

        key = tuple(r['parameters'][p] for p in group_by_params)

        if key not in summed_result:
            summed_result[key] = {m: {f: None for f in fields} for m in methods}
            key_count[key] = {m: {f: 0 for f in fields} for m in methods}

        for p in group_by_params:
            summed_result[key][method][p] = r['parameters'][p]

        for f in fields:
            if f in r:
                if summed_result[key][method][f] is None:
                    summed_result[key][method][f] = r[f]
                else:
                    summed_result[key][method][f] += r[f]

                key_count[key][method][f] += 1

    mean_result = {}

    for key in summed_result:
        mean_result[key] = {}

        for m in methods:
            mean_result[key][m] = {}

            for p in group_by_params:
                mean_result[key][m][p] = summed_result[key][m][p]

            for f in fields:
                if summed_result[key][m][f] is None:
                    mean_result[key][m][f] = None
                else:
                    mean_result[key][m][f] = summed_result[key][m][f] / key_count[key][m][f]

    return mean_result


def create_table(result, methods, params, fields):
    texts = [
        "\\begin{table}[htb]",
        "  \\begin{tabular}{c|" + "|".join(["r" * len(fields) for _ in methods]) + "}",
        "    & " + " & ".join(["\\multicolumn{" + str(len(fields)) + "}{c}{" + m + "}" for m in methods]) + " \\\\",
        "    " + (" & " + " & ".join(fields)) * len(methods) + " \\\\",
        "  \\hline"]

    for key in sorted(result.keys()):
        line = "    " + str(key)

        for m in methods:
            for f in fields:
                if f not in result[key][m] or result[key][m][f] is None:
                    line += " & -"
                else:
                    line += " & {:.2f}".format(result[key][m][f])

        line += " \\\\"
        texts.append(line)

    texts += ["    \\hline",
              "  \\end{tabular}",
              "\\end{table}"]

    return "\n".join(texts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--methods', '-m', type=str, nargs='+',
                        default=['CPLEX', 'gurobi', 'LNS'])
    parser.add_argument('--params', '-p', type=str, nargs='+',
                        default=['num_customers', 'num_perturbations'])
    parser.add_argument('--fields', '-f', type=str, nargs='+',
                        default=['cost', 'time_to_feasible', 'time_to_optimal'])
    parser.add_argument('--output', '-o', type=str, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        result = json.load(f)

    mean_result = average_result(result, args.methods, args.params, args.fields)
    table = create_table(mean_result, args.methods, args.params, args.fields)

    with open(args.output, 'w') as f:
        f.write(table + "\n")
