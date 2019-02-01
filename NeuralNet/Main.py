import csv
from random import randint

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import math
import time


def generate_etc_matrix(machines, tasks):
    new_etc = np.zeros(shape=(len(tasks), len(machines)), dtype=np.float64)

    for task_id in tasks.index.values:
        wl = tasks.values[task_id][0]

        for machine_id in machines.index.values:
            cc = machines.values[machine_id][0]
            new_etc[int(task_id)][int(machine_id)] = wl / cc

    return new_etc


def generate_individual(tasks, number_of_machines):
    number_of_tasks = len(tasks)
    tasks_per_machine = math.floor(number_of_tasks / number_of_machines)

    machines_chromosome = [tasks_per_machine] * number_of_machines
    for i in range(number_of_tasks - (tasks_per_machine * number_of_machines)):
        machines_chromosome[i] += 1
    return shuffle(tasks), shuffle(machines_chromosome)


def generate_population(p, tasks, machines):
    population = [None] * p
    number_of_machines = len(machines)
    for i in range(p):
        population += generate_individual(tasks, number_of_machines)
    return population


if __name__ == '__main__':
    try:

        machines = pd.read_csv("data/M-10.csv", skiprows=[0, 1], delimiter=";", index_col=[0], names=['id', 'CC'])
        tasks = pd.read_csv("data/T-200.csv", skiprows=[0, 1], delimiter=";", index_col=[0], names=['id', 'WL'])

        etc = generate_etc_matrix(machines, tasks)

    except KeyboardInterrupt:
        # niszczenie obiektow itp
        # (bezpieczne zamkniecie)
        pass
