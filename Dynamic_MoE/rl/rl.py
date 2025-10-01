import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional
DNA_SIZE = 24
POP_SIZE = 80
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.005
N_GENERATIONS = 100
MAX_EXPERT = 7
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

    def crossover_and_mutation(self, pops, CROSSOVER_RATE=0.8, MUTATION_RATE=0.01):
        new_pop = []
        pop_size = len(pops)
        for father in pops:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father.copy()  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pops[np.random.randint(pop_size)].copy()  # 再种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=0, high=self.DNA_size)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            child = self.mutation(child, MUTATION_RATE=MUTATION_RATE)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop
    
    def mutation(self, child, MUTATION_RATE=0.003):
        for mutate_point in range(len(child)):
            if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
                if np.random.rand() < 0.5:
                    if child[mutate_point] < MAX_EXPERT:
                        child[mutate_point] = child[mutate_point] + 1
                    else:
                    # 如果已达到最大值，则尝试减小
                        if child[mutate_point] > 0:
                            child[mutate_point] = child[mutate_point] - 1
                else:
                    if child[mutate_point] > 0:
                        child[mutate_point] = child[mutate_point] - 1
                    else:
                    # 如果已达到最小值，则尝试增大
                        if child[mutate_point] < MAX_EXPERT:
                            child[mutate_point] = child[mutate_point] + 1
        return child
    
    def select(self, elite_rate= 0.1, diversity_threshold = 0.3, cross_rate=0.8, mutation_rate=0.01):  # nature selection wrt pop's fitness
        inv_fitness = 1.0 / (np.array(self.fitness) ** DIFF_K)
        elite_count = max(1, int(self.POP_size * elite_rate))
        elite_indices = np.argsort(inv_fitness)[-elite_count:]
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
        else:
            selected_individuals = []
        # 对selected_individuals进行交叉变异
        selected_individuals = self.crossover_and_mutation(np.array(selected_individuals), cross_rate, mutation_rate )
        
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
    



if __name__ == "__main__":
    ga = GeneticAlgorithm(10, 24*3) # 16代表有16层，3代表每一层可以从000-111（二进制转化后为0-7）个专家中选择
    pop = ga.pop
    for _ in range(N_GENERATIONS):  # 迭代N代
        expert_list = ga.translateDNA()
        print(f"new pop is:\n {expert_list}")
        pop = np.array(ga.crossover_and_mutation(CROSSOVER_RATE))
        fitness = ga.get_fitness(np.random.uniform(low=1.0, high=5.0, size=ga.POP_size))
        pop = ga.select()  # 选择生成新的种群