import pandas as pd
import ray
from ray import tune
from ray.air import RunConfig
from ray.tune.schedulers import AsyncHyperBandScheduler

from lenet.train import train_model
from lenet.utils import load_yaml_config, get_project_root_dir


def tune_hyperparameters():
    ray.init(include_dashboard=False, num_cpus=4, num_gpus=1)
    project_root = get_project_root_dir()
    config_yaml = load_yaml_config("hyper_param_config.yaml")
    lr_low = float(config_yaml["lr"]["low"])
    lr_high = float(config_yaml["lr"]["high"])
    batch_size_list = config_yaml["batch_size"]["values"]
    map(int, batch_size_list)
    optimizer_list = config_yaml["optimizer"]["values"]
    num_epochs = int(config_yaml["epochs"]["value"])

    config = {
        "learning_rate": tune.loguniform(lr_low, lr_high),
        "batch_size": tune.grid_search(batch_size_list),
        "optimizer": tune.grid_search(optimizer_list),
        "project_root": project_root,
        "epochs": num_epochs
    }

    scheduler = AsyncHyperBandScheduler(time_attr="training_iteration")
    storage_path = f"{project_root}/test-results"

    tuner =  tune.Tuner(
        train_model,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=1,
            scheduler=scheduler,
            metric="eval_loss",  # Optimize for lowest evaluation loss
            mode="min" , # Minimize evaluation loss
        ),
        run_config=RunConfig(
            name="lenet_tune_experiment",
            storage_path=storage_path,
            log_to_file=True,
        )
    )

    results = tuner.fit()
    ray.shutdown()
    return results

def analyze_results(results):
    df = pd.DataFrame([result.metrics for result in results])
    print(df.columns)

    pareto_front = df[(df["eval_loss"] <= df["eval_loss"].min()) & (df["accuracy"] >= df["accuracy"].max()) & (df["time_total_s"] <= df["time_total_s"].min())]
    print("Pareto-optimal trials:", pareto_front)

if __name__ == "__main__":
    results = tune_hyperparameters()
    analyze_results(results)
