import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional
DNA_SIZE = 24
POP_SIZE = 80
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.01
N_GENERATIONS = 100
X_BOUND = [-2.048, 2.048]
Y_BOUND = [-2.048, 2.048]

class GeneticAlgorithm:
    def __init__(self, pop_size, length):
        self.DNA_size = length
        self.POP_size = pop_size
        self.fitness = None
        self.pop = np.random.randint(2, size=(self.POP_size, self.DNA_size))
    
    def F(self, x, label):
        return torch.nn.functional.cross_entropy(x, label)
    
    def translateDNA(self):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
        new_cols = self.DNA_size // 3
        reshaped_matrix = self.pop.reshape(self.POP_size, new_cols, 3)
        converted_matrix = np.packbits(reshaped_matrix, axis=-1) >> 5  # 右移5位以获取3位二进制数的整数值
        expert_list = np.squeeze(converted_matrix, axis=-1)
        print(expert_list)
        return expert_list
    
    def get_fitness(self, cross_entropy_list):
        """从外部获取，返回值是个torch.tensor，一维，长度为种群数目"""
        self.fitness = cross_entropy_list
        return self.fitness
        # return cross_entropy_list

    def crossover_and_mutation(self, CROSSOVER_RATE=0.8):
        new_pop = []
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pop[np.random.randint(self.POP_size)]  # 再种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=0, high=self.DNA_size)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            self.mutation(child)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop
    
    def mutation(self, child, MUTATION_RATE=0.003):
        if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, self.DNA_size)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转
    
    def select(self):  # nature selection wrt pop's fitness
        idx = np.random.choice(np.arange(self.POP_size), size=self.POP_size, replace=True,
                            p=(self.fitness) / (self.fitness.sum()))
        return pop[idx]
    
    def print_info(self):
        fitness = self.fitness
        max_fitness_index = np.argmax(fitness)
        print("max_fitness:", fitness[max_fitness_index])
        expert_list = self.translateDNA()
        print("最优的基因型：", self.pop[max_fitness_index])
        print("expert_list:", expert_list[max_fitness_index])
    
def F(x, y):
    return 100.0 * (y - x ** 2.0) ** 2.0 + (1 - x) ** 2.0  # 以香蕉函数为例


def get_fitness(pop):
    x, y = translateDNA(pop)
    pred = F(x, y)
    return pred
    # return pred - np.min(pred)+1e-3  # 求最大值时的适应度
    # return np.max(pred) - pred + 1e-3  # 求最小值时的适应度，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]


def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 0:DNA_SIZE]  # 前DNA_SIZE位表示X
    y_pop = pop[:, DNA_SIZE:]  # 后DNA_SIZE位表示Y

    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y


def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
    print(F(x[max_fitness_index], y[max_fitness_index]))

# def init_pop(expert_size, expert_len):
#     global POP_SIZE, DNA_SIZE
#     POP_SIZE = size
#     DNA_SIZE = length
    



if __name__ == "__main__":
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.ion()  # 将画图模式改为交互模式，程序遇到 plt.show 不会暂停，而是继续执行
    # pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    # for _ in range(N_GENERATIONS):  # 迭代N代
    #     x, y = translateDNA(pop)
    #     if 'sca' in locals():
    #         sca.remove()
    #     sca = ax.scatter(x, y, F(x, y), c='black', marker='o')
    #     plt.show()
    #     plt.pause(0.1)
    #     pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
    #     fitness = get_fitness(pop)
    #     pop = select(pop, fitness)  # 选择生成新的种群

    # print_info(pop)
    # plt.ioff()
    ga = GeneticAlgorithm(20, 16*3) # 16代表有16层，3代表每一层可以从000-111（二进制转化后为0-7）个专家中选择
    pop = ga.pop
    for _ in range(N_GENERATIONS):  # 迭代N代
        expert_list = ga.translateDNA()
        pop = np.array(ga.crossover_and_mutation(CROSSOVER_RATE))
        fitness = ga.get_fitness(torch.randint(low=1, high=11, size=(ga.POP_size,)))
        pop = ga.select()  # 选择生成新的种群