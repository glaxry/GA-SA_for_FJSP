#!/usr/bin/env python

# This script contains a high level overview of the proposed hybrid algorithm
# The code is strictly mirroring the section 4.1 of the attached paper
#!/usr/bin/env python

import sys
import time
from src.utils import parser, gantt
from src.genetic import encoding, decoding, genetic, termination
from src.simulatedannealing import simulated_annealing  # 确保您有这个模块和函数
from src import config

def update_population_with_optimized_individual(population, optimized_individual, index):
    # 替换优化后的个体
    population[index] = optimized_individual

def get_best_solution(population, parameters):
    # 获取最佳解决方案
    sorted_population = sorted(population, key=lambda ind: genetic.timeTaken(ind, parameters))
    return sorted_population[0]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <filename>")
        sys.exit()

    # 初始化
    t0 = time.time()
    parameters = parser.parse(sys.argv[1])
    population = encoding.initializePopulation(parameters)
    gen = 1

    while not termination.shouldTerminate(population, gen):
        # 遗传算法操作
        population = genetic.selection(population, parameters)
        population = genetic.crossover(population, parameters)
        population = genetic.mutation(population, parameters)

        # 模拟退火优化
        for index in range(config.top_individuals_for_SA):
            optimized_individual = simulated_annealing(population[index], parameters)
            update_population_with_optimized_individual(population, optimized_individual, index)

        gen += 1

    # 获取并显示最佳解决方案
    best_solution = get_best_solution(population, parameters)
    best_os, best_ms = best_solution

    t1 = time.time()
    total_time = t1 - t0
    print("Finished in {0:.2f}s".format(total_time))

    gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(parameters, best_os, best_ms))
    gantt.draw_chart(gantt_data)

