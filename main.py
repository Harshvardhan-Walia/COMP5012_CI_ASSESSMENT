#importing necessary libraries
import random
import matplotlib.pyplot as plt


#reading the data from the file and converting it to cost matrix.
def load_cost_matrix(filename):
    with open(filename, "r") as f:
        tokens = f.read().split()
    n = int(tokens[0])
    numbers = list(map(int, tokens[1:]))
    numbers = numbers[: n * n]
    cost = []
    idx = 0
    for _ in range(n):
        cost.append(numbers[idx: idx + n])
        idx += n

    return n, cost


#Function to create a valid solution also ensures no duplications
def make_random_assignment(n):
    assignment = list(range(n))
    random.shuffle(assignment)
    return assignment


#Calculates the values for one solution
def evaluate(assignment, cost):
    total_cost = 0
    max_cost = 0

    for worker, task in enumerate(assignment):
        c = cost[worker][task]
        total_cost += c
        if c > max_cost:
            max_cost = c

    return (total_cost, max_cost)


#Function to define Pareto Dominance
def dominates(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


#Class that rejects dominated solutions accepting only non dominated soln.
class ParetoArchive:

    def __init__(self):
        self.entries = []

    def add(self, obj, assignment):
        for p_obj, _ in self.entries:
            if dominates(p_obj, obj):
                return

        self.entries = [(p_obj, p_sol) for (p_obj, p_sol) in self.entries if not dominates(obj, p_obj)]
        self.entries.append((obj, assignment))


#Performs pure random search and generates and evaluates many random assignments
def random_search(n, cost, num_samples=10000):

    archive = ParetoArchive()
    all_points = []

    for _ in range(num_samples):
        assignment = make_random_assignment(n)
        obj = evaluate(assignment, cost)
        all_points.append(obj)
        archive.add(obj, assignment.copy())


    return all_points, archive


#Function to create a new solution from an existing solution also introduces small changes
def swap_mutation(assignment):

    a = assignment.copy()
    i, j = random.sample(range(len(a)), 2)
    a[i], a[j] = a[j], a[i]
    return a


#Scalarisation converting two objectives into a scalar value
def score_for_acceptance(obj, w):

    total_cost, max_cost = obj
    return w * total_cost + (1 - w) * (max_cost * 100)


#Optimiser
def simple_evolution(n, cost, iters=7000, weights=None):

    if weights is None:
        weights = [0.2, 0.5, 0.8]

    archive = ParetoArchive()

    for w in weights:

        current = make_random_assignment(n)
        current_obj = evaluate(current, cost)
        archive.add(current_obj, current.copy())

        for _ in range(iters):
            child = swap_mutation(current)
            child_obj = evaluate(child, cost)

            archive.add(child_obj, child.copy())

            if score_for_acceptance(child_obj, w) <= score_for_acceptance(current_obj, w):
                current = child
                current_obj = child_obj

    return archive


#Functions used to create plots
def plot_random(all_points, archive_points):
    X = [p[0] for p in all_points]
    Y = [p[1] for p in all_points]

    Ax = [p[0] for p in archive_points]
    Ay = [p[1] for p in archive_points]

    plt.figure()
    plt.scatter(X, Y, s=8, alpha=0.25, label="All random solutions")
    plt.scatter(Ax, Ay, s=50, label="Pareto archive (random)")
    plt.xlabel("Total cost (min)")
    plt.ylabel("Max individual cost (min)")
    plt.title("Assignment Problem: Random Search + Pareto Archive")
    plt.legend()
    plt.savefig("pareto_random_archive.png", bbox_inches="tight")
    plt.show()


def plot_evolution(archive_points):
    Ax = [p[0] for p in archive_points]
    Ay = [p[1] for p in archive_points]

    plt.figure()
    plt.scatter(Ax, Ay, s=60)
    plt.xlabel("Total cost (min)")
    plt.ylabel("Max individual cost (min)")
    plt.title("Pareto Archive after Simple Evolution swap mutation")
    plt.savefig("pareto_evolution_archive.png", bbox_inches="tight")
    plt.show()


#Main
if __name__ == "__main__":
    random.seed(42)
    print("\nLoaded dataset")
    n, cost = load_cost_matrix("assign100.txt")
    print("n =", n)
    print("Example costs :", cost[0][:10])

    print("\nRandom search")
    all_points, rand_archive = random_search(n, cost, num_samples=10000)
    print("Random archive size:", len(rand_archive.entries))

    print("\nSimple evolution")
    evo_archive = simple_evolution(n, cost, iters=7000, weights=[0.2, 0.5, 0.8])
    print("Evolution archive size:", len(evo_archive.entries))

    rand_objs = [obj for (obj, sol) in rand_archive.entries]
    evo_objs = [obj for (obj, sol) in evo_archive.entries]

    print("\nPlotting results ")
    plot_random(all_points, rand_objs)
    plot_evolution(evo_objs)

    print("Random Pareto points:", sorted(rand_objs))
    print("Evolution Pareto points:", sorted(evo_objs))


