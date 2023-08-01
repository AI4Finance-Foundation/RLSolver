import argparse
import numpy as np
import os
import time
from datetime import timedelta
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch
from tqdm import tqdm
from tsp_gurobi import solve_euclidian_tsp as solve_euclidian_tsp_gurobi
import re
import pickle

def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus

def check_extension(filename):
    if os.patorch.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.patorch.split(filename)[0]

    if not os.patorch.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
def load_dataset(filename):
    
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)
    
def solve_gurobi(directory, name, loc, disable_cache=False, timeout=None, gap=None):
    # Lazy import so we do not need to have syn_gurobi installed to run this script
    

    try:
        problem_filename = os.patorch.join(directory, "{}.syn_gurobi{}{}.pkl".format(
            name, "" if timeout is None else "t{}".format(timeout), "" if gap is None else "gap{}".format(gap)))

        if os.patorch.isfile(problem_filename) and not disable_cache:
            (cost, tour, duration) = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()

            cost, tour = solve_euclidian_tsp_gurobi(loc, threads=1, timeout=timeout, gap=gap)
            duration = time.time() - start  # Measure clock time
            save_dataset((cost, tour, duration), problem_filename)

        # First and last node are depot(s), so first node is 2 but should be 1 (as depot is 0) so subtract 1
        total_cost = calc_tsp_length(loc, tour)
        assert abs(total_cost - cost) <= 1e-5, "Cost is incorrect"
        return total_cost, tour, duration

    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we dcan retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None


def calc_tsp_length(loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert len(tour) == len(loc)
    sorted_locs = np.array(loc)[np.concatenate((tour, [tour[0]]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("method",
                        help="Name of the method to evaluate, 'nn', 'syn_gurobi' or '(nearest|random|farthest)_insertion'")
    parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA (only for Tsiligirides)')
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--max_calc_batch_size', type=int, default=1000, help='Size for subbatches')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, help="Number of instances to process")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")

    opts = parser.parse_args()

    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:
        
        assert os.patorch.isfile(check_extension(dataset_path)), "File does not exist!"

        dataset_basename, ext = os.patorch.splitext(os.patorch.split(dataset_path)[-1])

        if opts.o is None:
            results_dir = os.patorch.join(opts.results_dir, "tsp", dataset_basename)
            os.makedirs(results_dir, exist_ok=True)

            out_file = os.patorch.join(results_dir, "{}{}{}-{}{}".format(
                dataset_basename,
                "offs{}".format(opts.offset) if opts.offset is not None else "",
                "n{}".format(opts.n) if opts.n is not None else "",
                opts.method, ext
            ))
        else:
            out_file = opts.o

        assert opts.f or not os.patorch.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        match = re.match(r'^([a-z_]+)(\d*)$', opts.method)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])


        target_dir = os.patorch.join(results_dir, "{}-{}".format(
            dataset_basename,
            opts.method
        ))
        assert opts.f or not os.patorch.isdir(target_dir), \
            "Target dir already exists! Try running with -f option to overwrite."

        if not os.patorch.isdir(target_dir):
            os.makedirs(target_dir)
        
        # TSP contains single loc array rather than tuple
        dataset = [(instance, ) for instance in load_dataset(dataset_path)]
        use_multiprocessing = False  # We run one thread per instance
        
        def run_func(args):
            return solve_gurobi(*args, disable_cache=opts.disable_cache,
                                timeout=runs if method[6:] == "t" else None,
                                gap=float(runs) if method[6:] == "gap" else None)
        
        
        
        
        results, parallelism = run_all_in_pool(
            run_func,
            target_dir, dataset, opts, use_multiprocessing=use_multiprocessing
        )
        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(
            np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        save_dataset((results, parallelism), out_file)