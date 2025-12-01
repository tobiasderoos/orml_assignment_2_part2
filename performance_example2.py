# ---------------------------------------------------------------------
# Read the instance
# ---------------------------------------------------------------------
def read_instance(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    c = int(lines[1])
    w = list(map(int, lines[2].split()))
    p_lines = lines[3:]
    p = [list(map(int, row.split())) for row in p_lines]
    return n, c, w, p

# ---------------------------------------------------------------------
# Run the learning heuristic -- here something trivial
# ---------------------------------------------------------------------
def run_learningheuristic(instance_file):
    with open(instance_file, "r") as f:
        n, c, w, p = read_instance(instance_file)

    remcap = c
    alloc = [0 for i in range(n)]
    for i in range(n):
        if remcap >= w[i]:
            remcap = remcap - w[i]
            alloc[i] = 1

    solution_value = 0
    for i in range(n):
        for j in range(0,i+1):
            if alloc[i] * alloc[j] == 1:
                solution_value += p[i][j]
    return solution_value

# ---------------------------------------------------------------------
# Evaluate the subset of 20 instances and print the results in a file
# ---------------------------------------------------------------------
NUM_INSTANCES = 20
RESULTS_FILE = "results2.txt"

results = []

for i in range(NUM_INSTANCES):
    instance_file = f"InstancesEx2/instance{i}.txt"
    value = run_learningheuristic(instance_file)
    results.append(value)

with open(RESULTS_FILE, "w") as f:
    for val in results:
        f.write(f"{val}\n")

