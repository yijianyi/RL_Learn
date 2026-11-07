import multiprocessing as mp
import shlex

################################
# answer code
import importlib
################################


################################
# answer code
def _worker(job_str: str, entrypoint: str = "run"):
    job_args_list = shlex.split(job_str)
    assert job_args_list[0] == 'JOB'
    del job_args_list[0]  # Delete the dummy "JOB" prefix
    print(job_args_list)
    training = importlib.import_module(f"scripts.{entrypoint}")
    job_args = training.setup_arguments(args=job_args_list)
    training.main(job_args)
################################


################################
# answer code
def main_njobs(job_specs, njobs: int, start_method: str = "spawn", entrypoint: str = "run"):
    try:
        mp.set_start_method(start_method, force=True)
    except RuntimeError:
        pass

    with mp.Pool(processes=njobs) as pool:
        pool.starmap(_worker, [(spec, entrypoint) for spec in job_specs])
################################
