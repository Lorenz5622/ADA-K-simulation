from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional
DNA_SIZE = 24
POP_SIZE = 80
N_GENERATIONS = 100
MAX_EXPERT = 7
MAX_MUTATION = 0.5
DIFF_K = 4
np.set_printoptions(threshold=np.inf, linewidth=200)
class GeneticAlgorithm:
    def __init__(self, pop_size, length):
        self.DNA_size = length
        self.POP_size = pop_size
        self.fitness = None
        self.pop = np.random.randint(8, size=(self.POP_size, self.DNA_size))
    
    def F(self, x, label):
        return torch.nn.functional.cross_entropy(x, label)
    
    def translateDNA(self):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
        # print(expert_list)
        return self.pop
    
    def get_fitness(self, cross_entropy_list):
        """从外部获取，返回值是个torch.tensor，一维，长度为种群数目"""
        self.fitness = cross_entropy_list
        return self.fitness
        # return cross_entropy_list

    def crossover_and_mutation_old(self, pops, CROSSOVER_RATE=0.8, MUTATION_RATE=0.01):
        new_pop = []
        pop_size = len(pops)
        for father in pops:
            child = father.copy()
            if np.random.rand() < CROSSOVER_RATE:
                mother = pops[np.random.randint(pop_size)].copy()
                cross_points = np.random.randint(low=0, high=self.DNA_size)
                child[cross_points:] = mother[cross_points:]
            child = self.mutation(child, MUTATION_RATE=MUTATION_RATE)
            new_pop.append(child)

        return new_pop
    
    def crossover_and_mutation(self, pops, CROSSOVER_RATE=0.8, MUTATION_RATE=0.01):
        new_pop = []
        pop_size = len(pops)
        
        # 计算适应度值用于动态调整参数
        if self.fitness is not None:
            inv_fitness = 1.0 / (np.array(self.fitness) ** DIFF_K)
        else:
            inv_fitness = np.ones(pop_size)
        
        # 计算最大适应度和平均适应度
        max_inv_fitness = np.max(inv_fitness)
        avg_inv_fitness = np.mean(inv_fitness)
        
        for idx, father in enumerate(pops):
            child = father.copy()

            # 交叉操作
            if np.random.rand() < CROSSOVER_RATE:
                mother_idx = np.random.randint(pop_size)
                mother = pops[mother_idx].copy()
                
                # 计算交叉率
                if max_inv_fitness > avg_inv_fitness:  # 防止除零错误
                    # 取两个个体中适应度较高的那个
                    max_father_mother_fitness = max(inv_fitness[idx], inv_fitness[mother_idx])
                    if max_father_mother_fitness >= avg_inv_fitness:
                        individual_crossover_rate = CROSSOVER_RATE*(max_inv_fitness - max_father_mother_fitness) / (max_inv_fitness - avg_inv_fitness)
                    else:
                        individual_crossover_rate = CROSSOVER_RATE
                else:
                    individual_crossover_rate = CROSSOVER_RATE
                    
                # 应用交叉操作
                if np.random.rand() < individual_crossover_rate:
                    cross_points = np.random.randint(low=0, high=self.DNA_size)
                    child[cross_points:] = mother[cross_points:]

            # 计算个体变异率
            if max_inv_fitness > avg_inv_fitness:  # 防止除零错误
                
                if inv_fitness[idx] >= avg_inv_fitness:
                    individual_mutation_rate = MUTATION_RATE*(max_inv_fitness - inv_fitness[idx]) / (max_inv_fitness - avg_inv_fitness)
                else:
                    individual_mutation_rate = MUTATION_RATE
            else:
                individual_mutation_rate = MUTATION_RATE
            print(f"inv_fitness[idx]: {inv_fitness[idx]}, mutation rate: {individual_mutation_rate}")
                
            # 应用变异操作
            child = self.mutation(child, MUTATION_RATE=individual_mutation_rate)
            
            new_pop.append(child)

        return new_pop
    
    def mutation(self, child, MUTATION_RATE=0.003):
        for mutate_point in range(len(child)):
            if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
                if np.random.rand() < 0.5:
                    if child[mutate_point] < MAX_EXPERT:
                        child[mutate_point] = child[mutate_point] + 1
                    else:
                        if child[mutate_point] > 0:
                            child[mutate_point] = child[mutate_point] - 1
                else:
                    if child[mutate_point] > 0:
                        child[mutate_point] = child[mutate_point] - 1
                    else:
                        if child[mutate_point] < MAX_EXPERT:
                            child[mutate_point] = child[mutate_point] + 1
        return child
    
    def select(self, elite_rate= 0.1, diversity_threshold = 0.3, cross_rate=0.8, mutation_rate=0.01):  # nature selection wrt pop's fitness
        inv_fitness = 1.0 / (np.array(self.fitness) ** DIFF_K)
        elite_count = max(1, int(self.POP_size * elite_rate))
        elite_indices = np.argsort(inv_fitness)[:elite_count]
        elite_individuals = [self.pop[i].copy() for i in elite_indices]
        # print(elite_individuals)
        diversity = self.calculate_diversity()
        if diversity < diversity_threshold:
            # 在选择过程中增加随机性，避免过早收敛
            # 添加一些随机个体到选择池中
            random_count = max(1, int(self.POP_size * 0.1))
            random_indices = np.random.choice(np.arange(self.POP_size), size=random_count, replace=False)
            elite_indices = np.concatenate([elite_indices, random_indices])
            elite_indices = np.unique(elite_indices)
            elite_individuals = [self.pop[i].copy() for i in elite_indices]

        remaining_count = self.POP_size - len(elite_individuals)
        probs = inv_fitness / inv_fitness.sum()
        if remaining_count > 0:
            # 重新计算除去精英个体后的选择概率
            remaining_indices = np.setdiff1d(np.arange(self.POP_size), elite_indices)
            if len(remaining_indices) > 0:
                remaining_probs = probs[remaining_indices]
                remaining_probs = remaining_probs / remaining_probs.sum()  # 重新归一化
                
                # 选择剩余个体
                selected_indices = np.random.choice(remaining_indices, size=remaining_count, replace=True, p=remaining_probs)
                selected_individuals = [self.pop[i] for i in selected_indices]
            else:
                # 如果所有个体都是精英（罕见情况），则随机选择
                selected_indices = np.random.choice(np.arange(self.POP_size), size=remaining_count, replace=True)
                selected_individuals = [self.pop[i] for i in selected_indices]
                selected_fitness = [inv_fitness[i] for i in selected_indices]
        else:
            selected_individuals = []
            selected_fitness = []
        # 对selected_individuals进行交叉变异
        selected_individuals = self.crossover_and_mutation(
            np.array(selected_individuals), 
<<<<<<< HEAD
            fitness_values=np.array(selected_fitness) if selected_fitness else None,
            CROSSOVER_RATE=cross_rate, 
            base_mutation_rate=cross_rate,
            max_mutation_rate=MAX_MUTATION
=======
            CROSSOVER_RATE=cross_rate, 
            MUTATION_RATE=mutation_rate
>>>>>>> d525a148efc241e215a7508e373d0e817cf6f2bf
        )
        
        new_pop = elite_individuals + selected_individuals
        self.pop = np.array(new_pop[:self.POP_size])
        # print(self.pop)
        return self.pop
    
    
    def calculate_diversity(self):
        """
        计算种群多样性，通过计算个体间的平均汉明距离
        """
        if self.POP_size <= 1:
            return 0.0
        
        total_distance = 0
        count = 0
        
        # 计算所有个体对之间的汉明距离
        for i in range(self.POP_size):
            for j in range(i+1, self.POP_size):
                distance = np.sum(self.pop[i] != self.pop[j])
                total_distance += distance
                count += 1
        
        # 平均汉明距离除以基因长度，得到归一化的多样性度量
        avg_distance = total_distance / count if count > 0 else 0
        normalized_diversity = avg_distance / self.DNA_size
        
        return normalized_diversity
    
    def print_info(self):
        inv_fitness = 1.0 / (np.array(self.fitness) ** DIFF_K)
        print("best_fitness_original: ", (min(self.fitness)))
        # 归一化为概率
        fitness_norm = inv_fitness / inv_fitness.sum()
        max_fitness_index = np.argmax(fitness_norm)
        min_fitness_index = np.argmin(fitness_norm)
        print("max_fitness:", fitness_norm[max_fitness_index])
        print("min_fitness:", fitness_norm[min_fitness_index])
        expert_list = self.translateDNA()
        print("最优的基因型(max): ", expert_list[max_fitness_index])
        print("最优的基因型(min): ", expert_list[min_fitness_index])
    
    def write_to_record(self, file, gen):
        inv_fitness = 1.0 / (np.array(self.fitness) ** DIFF_K)
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        to_write = f"--------No. {gen}: {time}----------\n"
        to_write += (f"best_fitness_original: {min(self.fitness)} \n")
        # 归一化为概率
        fitness_norm = inv_fitness / inv_fitness.sum()
        max_fitness_index = np.argmax(fitness_norm)
        min_fitness_index = np.argmin(fitness_norm)
        to_write += (f"max_fitness: {fitness_norm[max_fitness_index]}\n")
        to_write += (f"min_fitness: {fitness_norm[min_fitness_index]}\n")
        expert_list = self.translateDNA()
        to_write += (f"最优的基因型(max):{expert_list[max_fitness_index]} \n")
        to_write += (f"最优的基因型(min): {expert_list[min_fitness_index]} \n")
        with open(file, 'a') as f:
            f.write(to_write)
        return
    
    

    



if __name__ == "__main__":
    ga = GeneticAlgorithm(10, 24*3) # 16代表有16层，3代表每一层可以从000-111（二进制转化后为0-7）个专家中选择
    pop = ga.pop
    for _ in range(N_GENERATIONS):  # 迭代N代
        expert_list = ga.translateDNA()
        print(f"new pop is:\n {expert_list}")
        pop = np.array(ga.crossover_and_mutation(CROSSOVER_RATE))
        fitness = ga.get_fitness(np.random.uniform(low=1.0, high=5.0, size=ga.POP_size))
        pop = ga.select()  # 选择生成新的种群