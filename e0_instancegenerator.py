import numpy as np
import random
import os


def generate_instance(n):
    weight = [random.randint(1, 100) for i in range(n)]
    profit = [[random.randint(1, 100) for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i, n):
            profit[j][i] = profit[i][j]
    cap = random.randint(round(0.4 * sum(weight)), round(0.6 * sum(weight)))

    return cap, weight, profit


if __name__ == "__main__":
    # Parameters
    n = 200
    seed = 9511
    np.random.seed(seed)
    dir = "InstancesEx1_test"
    os.makedirs(dir, exist_ok=True)

    for it in range(100):
        # Generate instance
        c, w, p = generate_instance(n)

        # Print instance to file
        filename = f"{dir}/instance{it}.txt"
        with open(filename, "w") as f:
            f.write(f"{n}\n")
            f.write(f"{c}\n")
            for i in w:
                f.write(f"{i}\t")
            f.write("\n")
            for i in range(n):
                for j in range(n):
                    f.write(f"{p[i][j]}\t")
                f.write("\n")
