import csv
from random import randint

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
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
    ret_tasks = shuffle(tasks.values)
    return [item for sublist in ret_tasks for item in sublist], shuffle(machines_chromosome)


def generate_population(p, tasks, machines):
    population = []
    number_of_machines = len(machines)
    for i in range(p):
        population.append(tuple(generate_individual(tasks, number_of_machines)))
    return population


def crossover(population):
    shuffled_population = shuffle(population)
    new_population = []
    for i in range(int(len(shuffled_population) / 2)):
        element = i * 2
        new_task_chromosome_a, new_task_chromosome_b = ordered_crossover(shuffled_population[element][0],
                                                                         shuffled_population[element + 1][0])

        new_population.append((new_task_chromosome_a, shuffled_population[element][1]))
        new_population.append((new_task_chromosome_b, shuffled_population[element + 1][1]))
    return new_population


def ordered_crossover(janusz, grazyna):
    size = len(grazyna)

    # Choose random start/end position for crossover
    karyna, seba = [-1] * size, [-1] * size
    start, end = sorted([random.randrange(size) for _ in range(2)])

    # Replicate grazyna's sequence for karyna, janusz's sequence for seba
    karyna_inherited = []
    seba_inherited = []
    for i in range(start, end + 1):
        karyna[i] = grazyna[i]
        seba[i] = janusz[i]
        karyna_inherited.append(grazyna[i])
        seba_inherited.append(janusz[i])

    # print(karyna, seba)
    # Fill the remaining position with the other parents' entries
    current_janusz_position, current_grazyna_position = 0, 0

    fixed_pos = list(range(start, end + 1))
    i = 0
    while i < size:
        if i in fixed_pos:
            i += 1
            continue

        test_karyna = karyna[i]
        if test_karyna == -1:  # to be filled
            janusz_trait = janusz[current_janusz_position]
            while janusz_trait in karyna_inherited:
                current_janusz_position += 1
                janusz_trait = janusz[current_janusz_position]
            karyna[i] = janusz_trait
            karyna_inherited.append(janusz_trait)

        # repeat block for seba and mom
        i += 1

    i = 0
    while i < size:
        if i in fixed_pos:
            i += 1
            continue

        test_seba = seba[i]
        if test_seba == -1:  # to be filled
            grazyna_trait = grazyna[current_grazyna_position]
            while grazyna_trait in seba_inherited:
                current_grazyna_position += 1
                grazyna_trait = grazyna[current_grazyna_position]
            seba[i] = grazyna_trait
            seba_inherited.append(grazyna_trait)

        # repeat block for seba and mom
        i += 1

    return karyna, seba


if __name__ == '__main__':
    try:

        machines = pd.read_csv("data/M-10.csv", skiprows=[0, 1], delimiter=";", index_col=[0], names=['id', 'CC'])
        tasks = pd.read_csv("data/T-200.csv", skiprows=[0, 1], delimiter=";", index_col=[0], names=['id', 'WL'])

        etc = generate_etc_matrix(machines, tasks)
        # crossover test;
        # x = generate_population(6, tasks, machines)
        # y = crossover(x)
        # z = crossover(y)

    except KeyboardInterrupt:
        # niszczenie obiektow itp
        # (bezpieczne zamkniecie)
        pass
