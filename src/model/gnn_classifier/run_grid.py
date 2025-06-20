import itertools
import copy
from classifier_config import classifier_config
from main import main
import traceback
from datetime import datetime


# Grille des paramètres à tester
grid = {
    "MLP_layers": [[],[128], [128, 64]],
    "classifier_encoder": [ "RGCN"],
    "num_bases": [5, 10],  # seulement utilisé si RGCN
    "encoder_out_channels": [[384,256], [256,256]]
}

keys, values = zip(*grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

log_file = "failed_runs.txt"

for i, exp_config in enumerate(experiments, 1):
    print(f"\n🚀 Running experiment {i}/{len(experiments)}: {exp_config}")
    config = copy.deepcopy(classifier_config)
    config.update(exp_config)

    if config["classifier_encoder"] != "RGCN":
        config["num_bases"] = None
    else:
        # Vérifie que c’est bien un int
        if isinstance(config["num_bases"], list):
            config["num_bases"] = config["num_bases"]

    try:
        main(config)
    except Exception as e:
        print(f" Experiment failed: {e}")
        with open(log_file, "a") as f:
            f.write("=" * 80 + "\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
            f.write(" Experiment failed with config:\n")
            for k, v in config.items():
                f.write(f"{k}: {v}\n")
            f.write("\nException:\n")
            f.write(traceback.format_exc())
            f.write("\n\n")