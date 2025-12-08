# To assess performance, you must also submit a self-contained Python file named
# performanceEx2.py that uses your trained model to run the deep Q-learning heuristic on a set of 20
# instances called instance0.txt, . . . , instance19.txt stored in the folder InstancesEx2, which we
# will generate with instanceGenerator.py using a different random seed. This file must produce an
# output file results2.txt containing one row per instance, where each row should contain a single
# number representing the solution value obtained by the deep Q-learning heuristic for that instance
# (see the file performance_example2.py for an example).
from e2_training_deep_q import load_agent, QEnv, QAgent


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
# Run the hyper-heuristic
# ---------------------------------------------------------------------
def run_hyperheuristic(instance_file, instance_id):
    with open(instance_file, "r") as f:
        n, cap, w, q = read_instance(instance_file)
    p = [q[i][i] for i in range(n)]
    env = QEnv(w, p, q, cap, instance_id=instance_id)

    # Load model
    loaded_agent = load_agent("exc_2_model/test_model.keras", env)
    # Evaluate agent on instance
    _, profit, _ = loaded_agent.evaluate(env)

    return profit


# ---------------------------------------------------------------------
# Evaluate the subset of 20 instances and print the results in a file
# ---------------------------------------------------------------------
NUM_INSTANCES = 10
RESULTS_FILE = "results2.txt"

results = []
for i in range(NUM_INSTANCES):
    instance_file = f"InstancesEx2/instance{i}.txt"
    value = run_hyperheuristic(instance_file, instance_id=i)
    results.append(value)

with open(RESULTS_FILE, "w") as f:
    for val in results:
        f.write(f"{val}\n")
