import random as rd
import numpy as np
import argparse
import sys
import os
import glob
import copy
import matplotlib.pyplot as plt
from datetime import datetime


class Agent:
    def __init__(self, n_items):
        self.items = np.array([float(rd.randint(0, 1)) for _ in range(n_items)])
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


#----- 定数部分を更新(a, A, C, l, p)
def update_const(a, a_step, n_items):
    #a, A, Cの更新
    a -= a_step
    r = np.random.uniform(0.0, 1.0, size=n_items)
    A = 2.0 * np.multiply(a, r) - a
    C = 2.0 * r
    #lの更新
    l = rd.uniform(-1.0, 1.0)
    #pの更新
    p = rd.uniform(0.0, 1.0)
    
    return a, A, C, l, p

def calc_encircle(sol, best_sol, A, C):
    D = np.linalg.norm(np.multiply(C, best_sol) - sol)
    return best_sol - np.multiply(A, D)

def calc_search(sol, rand_sol, A, C):
    D = np.linalg.norm(np.multiply(C, rand_sol) - sol)
    return rand_sol - np.multiply(A, D)

def calc_attack(sol, best_sol, l, b):
    D = np.linalg.norm(best_sol - sol)
    return np.multiply(np.multiply(D, np.exp(b * l)), np.cos(2.0*np.pi*l)) + best_sol


def adjust_agent(X_agents):
    for i in range(len(X_agents)):
        agent = X_agents[i]
        for j in range(len(agent.items)):
            if agent.items[j] < -1.0:
                agent.items[j] = -1.0
            elif agent.items[j] > 2.0:
                agent.items[j] = 2.0
        X_agents[i] = agent

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nsols", type=int, default=50, dest='nsols', help='number of solutions per generation, default: 50')
    parser.add_argument("-iter", type=int, default=100, dest='iter', help='number of iterations')
    parser.add_argument("-a", type=float, default=2.0, dest='a', help='woa algorithm specific parameter, controls search spread default: 2.0')
    parser.add_argument("-b", type=float, default=0.5, dest='b', help='woa algorithm specific parameter, controls spiral, default: 0.5')
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


def result(best, n_items, items, solution, solutions, capacity, x, y1, y2, problems, nsols):
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
    plt.ylim(min(y1) - 50, max(y2) + 50)
    plt.grid(linestyle='dotted')
    plt.legend(loc='lower right')
    
    date = datetime.now().strftime("%d_%H%M%S")
    
    fig.savefig(problems + "/iter"+ str(len(x)) + "_nsols" + str(nsols)+ "_"+date +".jpg")


def main():
    args = parse_cl_args()
    
    nsols = args.nsols
    n_iter = args.iter

    problems = args.problem
    capacity, items, n_items, solution, solutions= load_problems(problems)
    
    a = args.a
    a_step = a / n_iter / nsols
    a += a_step
    
    b = args.b
    
    x = [(i+1) for i in range(n_iter)]
    y1 = []
    y2 = [solution for _ in range(n_iter)]
    y_avg = []
    
    # 初期世代を生成
    X = initialize_population(nsols, n_items)
    
    # 適合度を計算
    for i in range(nsols):
        agent = X[i]
        agent.fitness = calc_fitness(agent, items, n_items, capacity)
        X[i] = agent
    
    # 一番適合度が高いエージェントを取り出す
    x_star = search_best_agent(X)
    
    # 世代間で繰り返し
    for t in range(n_iter):
        x_star_copy = copy.deepcopy(x_star)
        # １世代内で繰り返し
        for i in range(nsols):
            a, A, C, l, p = update_const(a, a_step, n_items)
            agent = X[i]
            agent_tmp = copy.deepcopy(agent)
            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    # 獲物に近づく
                    x_star_tmp = copy.deepcopy(x_star)
                    agent.items = calc_encircle(agent_tmp.items, x_star_tmp.items, A, C)
                else:
                    # 獲物を探す
                    rand_sol = rd.choice(X)
                    agent.items = calc_search(agent_tmp.items, rand_sol.items, A, C)
            else:
                # 回る
                x_star_tmp = copy.deepcopy(x_star)
                agent.items = calc_attack(agent_tmp.items, x_star_tmp.items, l, b)
            X[i] = agent

        # 適合度計算
        for i in range(nsols):
            agent = X[i]
            agent.fitness = calc_fitness(agent, items, n_items, capacity)
            X[i] = agent
        
        adjust_agent(X)

        # x_starを塗り替える
        x_star = copy.deepcopy(x_star_copy)
        max_agent = search_best_agent(X)
        if x_star.fitness < max_agent.fitness:
            x_star = max_agent
            for i in range(n_items):
                if x_star.items[i] > 0.5:
                    x_star.items[i] = 1.0
                else:
                    x_star.items[i] = 0.0

        y1.append(x_star.fitness)

        if t < n_iter - 1:
            print("iteration : " + str(t + 1) + ", best_value : " + str(x_star.fitness))

    #結果を出力
    result(x_star, n_items, items, solution, solutions, capacity, x, y1, y2, problems, nsols)


if __name__ == '__main__':
    main()