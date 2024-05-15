import numpy as np
from tqdm import tqdm


n_tasks = 28
n_persons = 14

tasks_per_person = 10
persons_per_task = 5

n_task_permutes = int(1e6)
n_trials = int(1e7)

binay_task_vector = np.zeros((n_tasks,), dtype=np.ubyte)
binay_task_vector[:tasks_per_person] = 1

non_zero_col_indices = []

for _ in tqdm(range(n_task_permutes)):
    non_zero_col_idx = np.random.choice(n_tasks, tasks_per_person)
    non_zero_col_indices.append(non_zero_col_idx)

binary_matrix = np.zeros((n_persons, n_tasks), dtype=np.ubyte)
selection_indices = []
for _ in tqdm(range(n_trials)):
    selection_idx = np.random.choice(n_task_permutes, n_persons)
    # selection_indices.append(non_zero_col_idx)

    for i, _idx in enumerate(selection_idx):
        binary_matrix[i, non_zero_col_indices[_idx]] = 1

    col_sum = np.count_nonzero(binary_matrix, axis=1)

    is_valid = np.all(col_sum == persons_per_task)

    if is_valid:
        print()
