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


# generowanie jednego osobnika
def generate_individual(tasks, number_of_machines):
    number_of_tasks = len(tasks)
    tasks_per_machine = math.floor(number_of_tasks / number_of_machines)

    # wypelnij macierz iloscia osobnikow na maszyne
    machines_chromosome = [tasks_per_machine] * number_of_machines
    # dodaj po jednym osobniku, jeśli ilość osobników na maszynę nie jest równa dla każdej maszyny
    for i in range(number_of_tasks - (tasks_per_machine * number_of_machines)):
        machines_chromosome[i] += 1
    ret_tasks = shuffle(tasks.values)
    # osobnik jest reprezentowany przez krotkę:
    # 1. lista z kolejnymi zadaniami (numer zadania)
    # 2. lista z iloscia zadan na maszynę (kolejnosc w liscie - numer maszyny)
    return [item for sublist in ret_tasks for item in sublist], shuffle(machines_chromosome)


# generowanie populacji
def generate_population(p, tasks, machines):
    population = []
    number_of_machines = len(machines)
    for i in range(p):
        population.append(tuple(generate_individual(tasks, number_of_machines)))
    return population


# krzyżowanie populacji
def crossover(population):
    # przemieszaj populację, tak aby brać losowe osobniki do krzyżowania, a nie kolejne
    shuffled_population = shuffle(population)
    new_population = []
    # bierz kolejno po dwa osobniki z populacji i krzyżuj ze sobą
    for i in range(int(len(shuffled_population) / 2)):
        element = i * 2
        new_task_chromosome_a, new_task_chromosome_b = ordered_crossover(shuffled_population[element][0],
                                                                         shuffled_population[element + 1][0])

        new_population.append((new_task_chromosome_a, shuffled_population[element][1]))
        new_population.append((new_task_chromosome_b, shuffled_population[element + 1][1]))
    return new_population


# krzyżowanie dwóch osobników
def ordered_crossover(dad, mom):
    # długość osobnika (ilość zadań)
    size = len(mom)

    # wybierz losową pozycje początku / końca krzyżowania
    daughter, son = [-1] * size, [-1] * size
    start, end = sorted([random.randrange(size) for _ in range(2)])

    # replikuj sekwencję matki dla córki i ojca dla syna
    daughter_inherited = []
    son_inherited = []
    for i in range(start, end + 1):
        daughter[i] = mom[i]
        son[i] = dad[i]
        daughter_inherited.append(mom[i])
        son_inherited.append(dad[i])

    # wypełnij pozostałe pozycje pozostałymi danymi z rodziców
    current_dad_position, current_mom_position = 0, 0

    fixed_pos = list(range(start, end + 1))
    i = 0
    while i < size:
        # pomiń już skrzyzowany fragment
        if i in fixed_pos:
            i += 1
            continue

        # wypełniaj pozostałe fragmenty
        test_daughter = daughter[i]
        if test_daughter == -1:  # wymaga wypelnienia
            dad_trait = dad[current_dad_position]
            while dad_trait in daughter_inherited:
                current_dad_position += 1
                dad_trait = dad[current_dad_position]
            daughter[i] = dad_trait
            daughter_inherited.append(dad_trait)

        test_son = son[i]
        if test_son == -1:  # wymaga wypelnienia
            mom_trait = mom[current_mom_position]
            while mom_trait in son_inherited:
                current_mom_position += 1
                mom_trait = mom[current_mom_position]
            son[i] = mom_trait
            son_inherited.append(mom_trait)
        i += 1

    return daughter, son


def get_highest_makespan(etc):
    machines_num = etc.shape[0]
    tasks_num = etc.shape[1]
    max_ms = 0
    for x in range(machines_num):
        current_ms = 0
        for y in range(tasks_num):
            current_ms += etc[y, x]
        if current_ms >= max_ms:
            max_ms = current_ms
    return max_ms


if __name__ == '__main__':
    try:

        machines = pd.read_csv("data/M-10.csv", skiprows=[0, 1], delimiter=";", index_col=[0], names=['id', 'CC'])
        tasks = pd.read_csv("data/T-200.csv", skiprows=[0, 1], delimiter=";", index_col=[0], names=['id', 'WL'])

        etc = generate_etc_matrix(machines, tasks)
        # crossover test;
        # x = generate_population(6, tasks, machines)
        # y = crossover(x)
        # z = crossover(y)
        # test for individual
        # a = [4, 9, 2, 8, 3, 1, 5, 7, 6]
        # b = [6, 4, 1, 3, 7, 2, 8, 5, 9]
        # print(ordered_crossover(a, b))

    except KeyboardInterrupt:
        # niszczenie obiektow itp
        # (bezpieczne zamkniecie)
        pass
