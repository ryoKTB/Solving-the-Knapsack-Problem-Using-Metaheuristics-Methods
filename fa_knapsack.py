import random as rd
import numpy as np
import argparse
import sys
import os
import glob
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import math


class Agent:
    def __init__(self, n_items):
        self.items = np.array([float(rd.randint(0, 1)) for _ in range(n_items)])
        self.fitness = 0

class Agent_sparks_mutation:
    def __init__(self, items):
        self.items = items
        self.fitness = 0

class Items:
    def __init__(self, weights, values):
        self.weights = weights
        self.values = values


#----- 初期状態を生成
def initialize_population(n_pop, n_items):
    return [Agent(n_items) for _ in range(n_pop)]

#----- 適応度を計算
def calc_fitness(agent, items, n_items, max_weight):
    value = 0
    weight = 0
    
    agent_items = agent.items
    weight_arr = items.weights
    value_arr = items.values
    
    for i in range(n_items):
        if agent_items[i] > 0.5:
            value += value_arr[i]
            weight += weight_arr[i]
            
    fitness = 0
    
    if weight > max_weight:
        fitness = -1
    else:
        fitness = value
    
    return fitness

#----- 適合度が最大の個体を探す
def search_best_agent(X_agents):
    return max(X_agents, key=lambda t: t.fitness)

# ０を１に、１を０に
def change_binary(input_num):
    if input_num == 0:
        return 1
    else:
        return 0

# 花火発火の際に使用する関数
def fireworks_explosion(agent, Step):
    n_items = len(agent.items)
    z = list(range(n_items))
    rd.shuffle(z)

    for i in range(Step):
        agent.items[z[i]] = change_binary(agent.items[z[i]])

    return agent.items

# 花火を起こす
def carry_out_fireworks(X_agents, nsols, Ac, items, n_items, capacity, a=0.8,  A_min=1, eps = 0.0001):
    ymin = 10**9
    ymax = -10**9
    sums = 0
    
    sparks = []
    
    for agent in X_agents:
        if ymin > agent.fitness:
            ymin = agent.fitness
        if ymax < agent.fitness:
            ymax = agent.fitness
        sums += agent.fitness

    for i in range(nsols):
        Ni = round(nsols * (X_agents[i].fitness - ymin + eps) / (sums - ymin + nsols + eps))
        if Ni < 1:
            Ni = 1
        elif Ni > a * nsols:
            Ni = round(a * nsols)
        Ai = A_min + math.floor(Ac * (ymax - X_agents[i].fitness + eps) / (nsols * ymax - sums + eps))
        
        Step = 0
        
        if Ai < 1:
            Step = 1
        elif Ai > Ac:
            Step = Ac
        else:
            Step = rd.choice([i for i in range(1, Ai + 1)])
        for j in range(Ni):
            spark_items = fireworks_explosion(X_agents[i], Step)
            agent_spark = Agent_sparks_mutation(spark_items)
            agent_spark.fitness = calc_fitness(agent_spark, items, n_items, capacity)
            sparks.append(agent_spark)

    return ymin, ymax, sums, sparks


# 突然変異分
def carry_out_mutation(X_agents, nsols, n_items, nm, best_agent, ymin, sums, items, capacity, Ac, A_min=1, eps = 0.0001):
    mutation_agents = []
    z = list(range(nsols))
    rd.shuffle(z)
    for i in range(nm):
        num = z[i]
        sj = []
        for j in range(n_items):
            if X_agents[num].items[j] == best_agent.items[j]:
                sj.append(j)

        Ai = A_min + math.floor(Ac * (X_agents[num].fitness - ymin + eps) / (sums - ymin * nsols + eps))

        Step_m = 1
        if Ai > Ac:
            Step_m = Ac
        elif Ai >= 1:
            Step_m = rd.choice([i for i in range(1, Ai + 1)])

        agent_mutation = Agent_sparks_mutation(fireworks_explosion(X_agents[num], Step_m))
        
        agent_mutation.fitness = calc_fitness(agent_mutation, items, n_items, capacity)
        
        mutation_agents.append(agent_mutation)
    
    return mutation_agents


# 二つの花火の距離を算出
def distance(items_i, items_j):
    s = 0
    for i in range(len(items_i)):
        if items_i[i] != items_j[i]:
            s += 1
    return s

# 次の世代にどのくらいの重みでもって返すかを決める
def prop(X_agents):
    nsols = len(X_agents)
    s = 0.000001
    d_arr = []
    for i in range(nsols):
        d_tmp = 0
        for j in range(nsols):
            d_tmp += distance(X_agents[i].items, X_agents[j].items)
        d_arr.append(d_tmp)
        s += d_tmp
    
    return [d_arr[i]/s for i in range(nsols)]


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nsols", type=int, default=50, dest='nsols', help='number of fireworks per generation, default: 50')
    parser.add_argument("-iter", type=int, default=100, dest='iter', help='number of iterations')
    parser.add_argument("-nm", type=int, default=5, dest='nm', help='number of mutation sparks')
    parser.add_argument("-p", type=str, default='./problems/07', dest='problem', help='Enter the directory containing the problem')

    args = parser.parse_args()
    return args


def load_problems(problems):
    print("____________________________________________")
    print("Now loading problems\n")
    if not os.path.exists(problems):
        raise ValueError("not exist such directory !")
    
    capacity_file = glob.glob(problems + "/*_c.txt")
    capacity_file = capacity_file[0]
    if not os.path.exists(capacity_file):
        raise ValueError("not exist capacity_file !")
    
    weights_file = glob.glob(problems + "/*_w.txt")
    weights_file = weights_file[0]
    if not os.path.exists(weights_file):
        raise ValueError("not exist weights_file !")
    
    profits_file = glob.glob(problems + "/*_p.txt")
    profits_file = profits_file[0]
    if not os.path.exists(profits_file):
        raise ValueError("not exist profits_file !")
    
    solution_file = glob.glob(problems + "/*_s.txt")
    solution_file = solution_file[0]
    if not os.path.exists(solution_file):
        raise ValueError("not exist solution_file !")
    
    capacity = 0
    weights = []
    profits = []
    solutions = []

    with open(capacity_file, 'r') as fc:
        file_data = fc.readlines()
        for line in file_data:
            capacity = int(line.rstrip())
    
    print("capacity : " + str(capacity) + "\n")
    
    with open(weights_file, 'r') as fw:
        file_data = fw.readlines()
        for line in file_data:
            weights.append(int(line.rstrip()))
    
    with open(profits_file, 'r') as fp:
        file_data = fp.readlines()
        for line in file_data:
            profits.append(int(line.rstrip()))
    
    with open(solution_file, 'r') as fs:
        file_data = fs.readlines()
        for line in file_data:
            solutions.append(int(line.rstrip()))
    
    print("items :")
    for i in range(1, len(weights) + 1):
        print("   " + str(i) + ". weight : " + str(weights[i-1]) +",  value : " +  str(profits[i-1]))
    print("")
    
    items = Items(weights, profits)
    
    solution = 0
    for i in range(len(weights)):
        if solutions[i] == 1:
            solution += profits[i]
    
    print("____________________________________________")
    
    return capacity, items, len(weights), solution, solutions


def result(best, n_items, items, solution, solutions, capacity, x, y1, y2, problems, nsols, nm):
    print("____________________________________________")
    print("Result")
    print("")
    nums = []
    weights = items.weights
    values = items.values
    labels = best.items
    total_weight = 0
    total_values = 0
    
    for i in range(n_items):
        if labels[i] > 0.5:
            nums.append(i)
            total_weight += weights[i]
            total_values += values[i]
    
    print("total weights : " + str(total_weight) + " (capacity : " + str(capacity) + "),   total values : " + str(total_values))
    for i in nums:
        print("   " + str(i + 1) + ". weight : " + str(weights[i]) + ",  value : " + str(values[i]))
    print("")

    print("※ Assumed solution : " + str(solution))
    for i in range(n_items):
        if solutions[i] == 1:
            print("   " + str(i + 1) + ". weight : " + str(weights[i]) + ",  value : " + str(values[i]))
    print("")
    
    fig = plt.figure()
    fig.subplots_adjust(left=0.2)
    plt.plot(x, y1, linestyle='solid', color='blue', label="best_fitness")
    plt.plot(x, y2, linestyle='dashdot', color='red', label='assumed_solution')
    plt.title("Best Fitness")
    plt.xlabel("iteration")
    plt.ylabel("best fitness")
    plt.ylim(min(y1) - 30, max(y2) + 30)
    plt.grid(linestyle='dotted')
    plt.legend(loc='lower right')
    
    date = datetime.now().strftime("%d_%H%M%S")
    
    fig.savefig(problems + "/fa_nsols"+ str(nsols) + "_iter" + str(len(x))+ "_nm" + str(nm) + "_" +date +".jpg")


def main():
    args = parse_cl_args()

    nsols = args.nsols
    n_iter = args.iter
    nm = args.nm

    problems = args.problem
    capacity, problem_items, n_items, solution, solutions= load_problems(problems)

    Ac = min(round(n_items * 0.8), round(nsols * 0.5))

    x = [(i+1) for i in range(n_iter)]
    y1 = []
    y2 = [solution for _ in range(n_iter)]

    # 初期世代を生成
    X = initialize_population(nsols, n_items)

    # 適合度を計算
    for i in range(nsols):
        agent = X[i]
        agent.fitness = calc_fitness(agent, problem_items, n_items, capacity)
        X[i] = agent

    best_agent = search_best_agent(X)

    # 世代間で繰り返し
    for t in range(n_iter):

        # 花火を起こす
        X_now = copy.deepcopy(X)
        ymin, ymax, sums, sparks = carry_out_fireworks(X_now, nsols, Ac, problem_items, n_items, capacity)

        # ”突然変異”の分を計算
        X_now = copy.deepcopy(X)
        mutations = carry_out_mutation(X_now, nsols, n_items, nm, best_agent, ymin, sums, problem_items, capacity, Ac)

        X_new = sparks + mutations + X

        # 適合度を改めて計算
        for i in range(len(X_new)):
            agent_x = X_new[i]
            agent_x.fitness = calc_fitness(agent_x, problem_items, n_items, capacity)
            X_new[i] = agent_x

        best_agent = search_best_agent(X_new)
        X_new.remove(best_agent)

        # 一番適合度の高いものは次の世代に必ず残し、それ以外は指定された確率(prop関数で決める)に従って残す（エリート戦略）
        X = rd.choices(X_new, k=nsols-1, weights=prop(X_new))
        X.append(best_agent)

        y1.append(best_agent.fitness)

        if t < n_iter - 1:
            print("iteration : " + str(t + 1) + ", best_value : " + str(best_agent.fitness))

        #結果を出力
        if t == n_iter - 1:
            result(X[-1], n_items, problem_items, solution, solutions, capacity, x, y1, y2, problems, nsols, nm)


if __name__ == '__main__':
    main()