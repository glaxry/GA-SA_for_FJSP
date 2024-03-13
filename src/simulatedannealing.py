from src.genetic.genetic import timeTaken, selection
import random
import math
from src.genetic import decoding

def acceptance_probability(current_cost, candidate_cost, temperature):
    if candidate_cost < current_cost:
        return 1.0
    else:
        return math.exp((current_cost - candidate_cost) / temperature)


def generate_neighbor(solution, parameters):
    OS, MS = solution
    neighbor_OS = list(OS)
    neighbor_MS = list(MS)

    # 选择邻域搜索类型：MS变化还是OS变化
    if random.choice([True, False]):
        # MS变化：随机选择一个操作，改变其机器选择
        op_index = random.randint(0, len(neighbor_MS) - 1)
        job_idx, op_idx_within_job = decoding.decode_operation_index(op_index, parameters)
        available_machines = parameters['jobs'][job_idx][op_idx_within_job]
        new_machine_idx = random.randint(0, len(available_machines) - 1)
        neighbor_MS[op_index] = new_machine_idx
    else:
        # OS变化：随机交换两个操作的顺序
        op1_index, op2_index = random.sample(range(len(neighbor_OS)), 2)
        neighbor_OS[op1_index], neighbor_OS[op2_index] = neighbor_OS[op2_index], neighbor_OS[op1_index]

    return neighbor_OS, neighbor_MS


def simulated_annealing(init_solution, parameters, initial_temperature =1000, cooling_rate=0.95, final_temperature=1):
    current_solution = init_solution
    current_cost = timeTaken(current_solution, parameters)
    temperature = initial_temperature

    while temperature > final_temperature:
        candidate_solution = generate_neighbor(current_solution, parameters)
        candidate_cost = timeTaken(candidate_solution, parameters)

        if acceptance_probability(current_cost, candidate_cost, temperature) > random.random():
            current_solution = candidate_solution
            current_cost = candidate_cost

        temperature *= cooling_rate

    return current_solution

# 在GA的每一代结束后，对选出的个体应用SA进行优化
def optimize_with_sa(population, parameters):
    optimized_population = []
    for individual in selection(population, parameters):  # 使用遗传算法中的选择函数选择个体
        optimized_individual = simulated_annealing(individual, parameters, initial_temperature=1000, cooling_rate=0.95, final_temperature=1)
        optimized_population.append(optimized_individual)
    return optimized_population
