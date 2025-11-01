import numpy as np
random_pop = np.random.randint(7, size=(3, 8)).tolist()
pop = np.random.randint(7, size=(3, 8)).tolist()
pop += random_pop
print(pop)