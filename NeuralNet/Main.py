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

def mutate_population(p):
    for i in range(int(len(p))):       
        if check_swap_mutation():
            p[i] = swap_mutation(p[i]) 
        if check_transposition_mutation():
            p[i] = transposition_mutation(p[i])
    return p
        

def check_swap_mutation():
    random_value = np.random.uniform(0.0, 1.0)
    pms =0.1
    if random_value <=  pms:
        return 1
    else:
        return 0

def check_transposition_mutation():
    random_value = np.random.uniform(0.0, 1.0)
    pms =0.05
    if random_value <=  pms:
        return 1
    else:
        return 0

def swap_mutation(individual):
    number_of_tasks = len(individual[0]) -1
    i = np.random.randint(0, number_of_tasks)
    j = np.random.randint(0, number_of_tasks)
    while i == j:
        j = np.random.randint(0, number_of_tasks)
    pom = individual[0][i]
    individual[0][i] = individual[0][j]
    individual[0][j] = pom
    return individual
  

def transposition_mutation(individual):
    number_of_tasks = len(individual[0]) -1
    i = np.random.randint(0, number_of_tasks)
    j = np.random.randint(0, number_of_tasks)
    while i == j:
            j = np.random.randint(0, number_of_tasks)     
    
    position_to_change = get_machine_number_for_task(individual,i)
    if individual[1][position_to_change] <= 1:
        return individual
    destination_to_change = get_machine_number_for_task(individual,j)
    individual[1][position_to_change] = individual[1][position_to_change] -1 
    individual[1][destination_to_change] = individual[1][destination_to_change] + 1
    if i<j:
        traspositioned_number = individual[0][i]
        for x in range(i,j):
            individual[0][x] = individual[0][x+1]
        individual[0][j] ==transpositioned_number  
    else:
        traspositioned_number = individual[0][j]
        for x in reversed(range(j+1,i+1)):
            individual[0][x] = individual[0][x-1]
        individual[0][i] ==transpositioned_number
    return individual  

def get_machine_number_for_task(individual, task_position):
    counter = 0
    position_number = 0
    for machine in individual[1]:
        position_number += machine
        if position_number >= task_position+1:
            return counter
        else:
            counter = counter + 1
    return null

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


def get_highest_makespan_for_individual(etc, individual):
    task_arr = #nie wiem jak wyciąnąc tablice z numerami zadan z osobnika
    machine_arr = #podobnie nie wiem jak wyciągnąc tablice z ilością zadań na maszynie
    offset = 0
    max_makespan = 0
    for m in range(machine_arr): #m to id maszyny
        current_makespan = 0
        num_of_tasks = machine_arr[m] #na pozycji m jest ilosc zadan
        for t in range(num_of_tasks):
            index = t + offset
            current_makespan += etc[index][m] #dodajemy czas dla zadan dla maszyny m
        if current_makespan > max_makespan:
            max_makespan = current_makespan
        offset = range(num_of_tasks) #offset aby brac dalsze elementy tablicy z zadaniami
    return max_makespan


if __name__ == '__main__':
    try:

        machines = pd.read_csv("data/M-10.csv", skiprows=[0, 1], delimiter=";", index_col=[0], names=['id', 'CC'])
        tasks = pd.read_csv("data/T-200.csv", skiprows=[0, 1], delimiter=";", index_col=[0], names=['id', 'WL'])

        etc = generate_etc_matrix(machines, tasks)
        for x in reversed(range(20+1,30+1)):
            print(x)
        
        #crossover test;
        x = (6, tasks, machines)
        y = crossover(x)
        y = mutate_population(y)
        z = crossover(y)
        # test for individual
        # a = [4, 9, 2, 8, 3, 1, 5, 7, 6]
        # b = [6, 4, 1, 3, 7, 2, 8, 5, 9]
        # print(ordered_crossover(a, b))

    except KeyboardInterrupt:
        # niszczenie obiektow itp
        # (bezpieczne zamkniecie)
        pass
