import os
import sys

import numpy as np
from tqdm import tqdm
from datetime import datetime


def count_pairwise_assignments(binary_matrix):
    n_persons, n_tasks = binary_matrix.shape
    n_pairwise_assignments = {}
    n_pairwise_assignments_list = []

    for person_1 in range(n_persons):
        for person_2 in range(person_1 + 1, n_persons):
            pairwise_tasks = [k for k in range(n_tasks) if binary_matrix[person_1, k] and binary_matrix[person_2, k]]
            n_pairwise_assignments[(person_1, person_2)] = len(pairwise_tasks)
            n_pairwise_assignments_list.append(len(pairwise_tasks))

    avg_pairwise_assignments = np.mean(n_pairwise_assignments_list)
    return n_pairwise_assignments, avg_pairwise_assignments


def save_matrix(binary_matrix, unique_values, unique_counts, curr_0_count, curr_01_count, prefix, init_id,
                generation_id):
    unique_counts_str = '__'.join('{}-{}'.format(val, cnt) for val, cnt in zip(unique_values, unique_counts))
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_fname = '{}_init_{}_gen_{}___{}___01-{}___{}.csv'.format(
        prefix, init_id, generation_id, unique_counts_str, curr_01_count, time_stamp)
    np.savetxt(os.path.join('log', out_fname), binary_matrix, fmt='%d', delimiter='\t')

    return out_fname


def generate_random_assignment_matrix(n_persons, n_tasks, persons_per_task, tasks_per_person_list):
    person_to_n_tasks = np.zeros((n_persons,), dtype=np.ubyte)
    available_persons = list(range(n_persons))
    assignment_matrix = np.zeros((n_persons, n_tasks), dtype=np.ubyte)
    tasks_per_person_list_rand = np.random.permutation(tasks_per_person_list)

    for task_id in range(n_tasks):
        available_persons = [k for k in available_persons if person_to_n_tasks[k] < tasks_per_person_list_rand[k]]
        n_available_persons = len(available_persons)

        if n_available_persons < persons_per_task:
            # print('Ran out of available_persons in task_id: {}'.format(task_id + 1))
            return None

        person_idx = np.random.permutation(available_persons)[:persons_per_task]

        for i, _idx in enumerate(person_idx):
            person_to_n_tasks[_idx] += 1

            assignment_matrix[_idx, task_id] = 1

    return assignment_matrix


def get_metrics(binary_matrix):
    n_persons, n_tasks = binary_matrix.shape

    n_pairwise_assignments, avg_pairwise_assignments = count_pairwise_assignments(binary_matrix)

    n_pairwise_assignments_list = list(n_pairwise_assignments.values())
    unique_values, unique_counts = np.unique(n_pairwise_assignments_list, return_counts=True)

    unique_values = list(unique_values)
    unique_counts = list(unique_counts)

    curr_0_count = 0
    if 0 in unique_values:
        curr_0_count = unique_counts[unique_values.index(0)]

    curr_1_count = 0
    if 1 in unique_values:
        curr_1_count = unique_counts[unique_values.index(1)]

    return unique_values, unique_counts, avg_pairwise_assignments, curr_0_count, curr_1_count


def main():
    method = 0
    n_tasks = 9
    type_1_tasks = 5
    n_persons = 14
    task_1_per_person = 1
    tasks_per_person = 2
    persons_per_task = 3

    allow_partial_decrease = 1

    n_trials = int(1e7)
    max_gen_trials = int(1e7)

    init_trials = 0
    gen_init = 1

    load_init = ''

    # load_init = 'log/max_3_count_init_33_gen_34___2-1__3-82__4-8___dev-4___210809_152458.csv'
    # load_init = 'log/max_3_count_init_241_gen_37__2-2__3-81__4-7__5-1__dev_2__210810_002125.csv'
    # load_init = 'evo_max_3_count___1-1__2-5__3-76__4-4__5-5___210805_181606.csv'

    cmd_args = sys.argv[1:]
    cmd_id = 0

    n_cmd_args = len(cmd_args)

    print('cmd_args: {}'.format(cmd_args))
    print('n_cmd_args: {}'.format(n_cmd_args))

    if n_cmd_args > cmd_id:
        max_gen_trials = int(float(cmd_args[cmd_id]))
        cmd_id += 1

    if n_cmd_args > cmd_id:
        load_init = cmd_args[cmd_id]
        cmd_id += 1

    type_2_tasks = n_tasks - type_1_tasks
    task_2_per_person = tasks_per_person - task_1_per_person

    if load_init:
        assert os.path.isfile(load_init), "non-existent load_init: {}".format(load_init)
        print('loading initial matrix from: {}'.format(load_init))
        binary_matrix = np.loadtxt(load_init)

        assert binary_matrix.shape == (n_persons, n_tasks), "loaded matrix has invalid shape"

        binary_matrix_type_1 = binary_matrix[:, :type_1_tasks]
        binary_matrix_type_2 = binary_matrix[:, type_1_tasks:]

        gen_init = 0

    global_max_0_count = 0
    global_max_01_count = 0

    global_min_deviations = np.inf
    prefix = 'evo'
    init_id = 0
    out_fname = None

    col_idx = list(range(n_tasks))
    row_idx = list(range(n_persons))

    os.makedirs('log', exist_ok=True)
    tasks_per_person_list = [2, ] * (n_persons - 1)
    tasks_per_person_list.append(1)

    type_1_tasks_per_person_list = [1, ] * (n_persons - 1) + [2, ]
    type_2_tasks_per_person_list = [0, ] * 2 + [1, ] * (n_persons - 2)

    all_tasks = list(range(n_tasks))
    while True:

        if gen_init:
            gen_init = 0
            init_id += 1

            print('searching for an initial random parent matrix...')

            while True:
                init_trials += 1

                if method == 0:

                    binary_matrix_type_1 = generate_random_assignment_matrix(n_persons, type_1_tasks,
                                                                             persons_per_task,
                                                                             type_1_tasks_per_person_list)
                    if binary_matrix_type_1 is None:
                        continue

                    # binary_matrix_type_1 = binary_matrix_type_1[:n_persons, :]

                    # col_sum_type_1 = np.count_nonzero(binary_matrix_type_1, axis=1)

                    # print('col_sum_type_1: {}'.format(col_sum_type_1))

                    # is_valid_type_1 = np.all(col_sum_type_1 == task_1_per_person)

                    # if not is_valid_type_1:
                    #     continue

                    binary_matrix_type_2 = generate_random_assignment_matrix(n_persons, type_2_tasks,
                                                                             persons_per_task,
                                                                             type_2_tasks_per_person_list)

                    # binary_matrix_type_2 = binary_matrix_type_2[:n_persons, :type_2_tasks]

                    # col_sum_type_2 = np.count_nonzero(binary_matrix_type_2, axis=1)

                    # n_task_2_assigns = np.count_nonzero(col_sum_type_2 == task_2_per_person)
                    # is_valid_type_2 = n_task_2_assigns == n_persons - 1

                    # print('col_sum_type_2: {}'.format(col_sum_type_2))
                    # print('n_task_2_assigns: {}'.format(n_task_2_assigns))

                    # if not is_valid_type_2:
                    #     continue

                    binary_matrix = np.concatenate((binary_matrix_type_1, binary_matrix_type_2), axis=1)

                    col_sum = np.count_nonzero(binary_matrix, axis=1)

                    # print('col_sum: {}'.format(col_sum))
                    # print()

                    n_assigns = np.count_nonzero(col_sum == tasks_per_person)
                    is_valid = n_assigns == n_persons - 1

                    if not is_valid:
                        continue

                elif method == 1:

                    binary_matrix = generate_random_assignment_matrix(n_persons, n_tasks, persons_per_task,
                                                                      tasks_per_person_list)
                    if binary_matrix is None:
                        continue

                    while True:
                        all_tasks_rand = np.random.permutation(all_tasks)

                        type_1_ids = all_tasks_rand[:type_1_tasks]
                        type_2_ids = all_tasks_rand[type_1_tasks:]

                        binary_matrix_type_1 = binary_matrix[:, type_1_ids]
                        binary_matrix_type_2 = binary_matrix[:, type_2_ids]

                        col_sum_type_1 = np.count_nonzero(binary_matrix_type_1, axis=1)
                        col_sum_type_2 = np.count_nonzero(binary_matrix_type_2, axis=1)

                        # print('col_sum: {}'.format(col_sum))
                        # print('col_sum_type_1: {}'.format(col_sum_type_1))
                        # print('col_sum_type_2: {}'.format(col_sum_type_2))
                        # print()

                        is_valid_type_1 = np.all(col_sum_type_1 == 1)

                        if not is_valid_type_1:
                            continue

                        n_task_2_assigns = np.count_nonzero(col_sum_type_2 == 1)

                        is_valid_type_2 = n_task_2_assigns == n_persons - 1

                        if not is_valid_type_2:
                            continue
                break

            print('initialization {} completed in {} trials'.format(init_id, init_trials))

        unique_values, unique_counts, avg_pairwise_assignments, \
        curr_0_count, curr_1_count = get_metrics(binary_matrix)

        curr_01_count = curr_0_count + curr_1_count

        print('unique_values:  {}'.format(unique_values))
        print('unique_counts:  {}'.format(unique_counts))
        print('avg_pairwise_assignments:  {}'.format(avg_pairwise_assignments))
        print()

        # gen_init = 1
        # continue

        max_0_count = curr_0_count
        max_01_count = curr_01_count

        # min_deviations = curr_deviations

        child_trials = 0
        generation_trials = 0
        generation_id = 0

        save = 0

        if max_0_count > global_max_0_count:
            global_max_0_count = max_0_count
            prefix = 'max_0_count'
            save = 1

        if max_01_count > global_max_01_count:
            global_max_01_count = max_01_count
            prefix = 'max_01_count'
            save = 1

        # if min_deviations < global_min_deviations:
        #     global_min_deviations = min_deviations
        #     prefix = 'min_deviations'
        #     save = 1

        if save:
            out_fname = save_matrix(binary_matrix, unique_values, unique_counts,
                                    curr_0_count, curr_01_count, prefix, init_id,
                                    generation_id)

        parent_binary_matrix = binary_matrix.copy()

        print('max_gen_trials:  {:e}'.format(max_gen_trials))
        # print('global_min_deviations:  {}'.format(global_min_deviations))
        print('global_max_0_count:  {}'.format(global_max_0_count))
        print('global_max_01_count:  {}'.format(global_max_01_count))

        print('out_fname:  {}'.format(out_fname))

        while True:
            generation_trials += 1
            child_trials += 1

            if generation_trials > max_gen_trials > 0:
                gen_init = 1
                break

            x1, x2 = np.random.permutation(col_idx)[:2]
            y1, y2 = np.random.permutation(row_idx)[:2]

            a, b, c, d = parent_binary_matrix[y1, x1], \
                         parent_binary_matrix[y1, x2], \
                         parent_binary_matrix[y2, x1], \
                         parent_binary_matrix[y2, x2]

            if a != d or a == b or b != c:
                continue

            # print('\nvalid child found in {} trials'.format(child_trials))

            child_trials = 0

            binary_matrix = parent_binary_matrix.copy()
            binary_matrix[y1, x1] = 1 - binary_matrix[y1, x1]
            binary_matrix[y2, x1] = 1 - binary_matrix[y2, x1]
            binary_matrix[y1, x2] = 1 - binary_matrix[y1, x2]
            binary_matrix[y2, x2] = 1 - binary_matrix[y2, x2]

            binary_matrix_type_1 = binary_matrix[:, :type_1_tasks]
            binary_matrix_type_2 = binary_matrix[:, type_1_tasks:]

            col_sum_type_1 = np.count_nonzero(binary_matrix_type_1, axis=1)
            col_sum_type_2 = np.count_nonzero(binary_matrix_type_2, axis=1)

            col_sum_type_1_sorted = sorted(list(col_sum_type_1))

            is_valid_type_1 = col_sum_type_1_sorted == type_1_tasks_per_person_list
            if not is_valid_type_1:
                continue

            col_sum_type_2_sorted = sorted(list(col_sum_type_2))
            is_valid_type_2 = col_sum_type_2_sorted == type_2_tasks_per_person_list
            if not is_valid_type_2:
                continue

            # print('\n\ninit {} valid generation found in {} trials'.format(
            #     init_id, generation_id, generation_trials))

            unique_values, unique_counts, avg_pairwise_assignments, \
            curr_0_count, curr_1_count = get_metrics(binary_matrix)
            curr_01_count = curr_0_count + curr_1_count

            if not allow_partial_decrease and (curr_0_count < max_0_count or curr_01_count < max_01_count):
                continue

            if curr_0_count <= max_0_count and curr_01_count <= max_01_count:
                continue

            # row_sum = np.count_nonzero(binary_matrix, axis=0)
            # col_sum = np.count_nonzero(binary_matrix, axis=1)
            # is_valid = np.all(col_sum == tasks_per_person) and np.all(row_sum == persons_per_task)
            # assert is_valid, "invalid assignment matrix generated even by performing valid operations"

            generation_id += 1

            print('\n\ninit {} improved generation {} found in {} trials'.format(
                init_id, generation_id, generation_trials))

            print('curr_0_count:  {}'.format(curr_0_count))
            print('curr_01_count:  {}'.format(curr_01_count))

            generation_trials = 0

            if curr_0_count > max_0_count:
                max_0_count = curr_0_count

            if curr_01_count > curr_01_count:
                curr_01_count = curr_01_count

            parent_binary_matrix = binary_matrix.copy()

            save = 0
            if curr_0_count > global_max_0_count:
                global_max_0_count = curr_0_count
                prefix = 'max_0_count'
                save = 1

            if curr_01_count > global_max_01_count:
                global_max_01_count = curr_01_count
                prefix = 'max_01_count'
                save = 1

            if save:
                out_fname = save_matrix(binary_matrix, unique_values, unique_counts,
                                        curr_0_count, curr_01_count,
                                        prefix, init_id, generation_id)

            print('global_max_0_count:  {}'.format(global_max_0_count))
            print('global_max_01_count:  {}'.format(global_max_01_count))
            print('out_fname:  {}'.format(out_fname))


if __name__ == '__main__':
    main()
