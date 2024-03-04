import os, subprocess, time, sys
import warnings
import pyRAPL
import pandas as pd
import numpy as np

from test import main as main_test

warnings.filterwarnings("ignore")


def energy_measurement(path_to_checkpoint_file):
    # Number of times test.py will be run to average the energy consumption
    n_measurements = 1

    # Setting up the pyRAPL library and choosing the measurements outputs
    pyRAPL.setup()
    output_test = pyRAPL.outputs.DataFrameOutput()  # save the outputs in a dataframe
    output_base = pyRAPL.outputs.DataFrameOutput()

    # Measure test.py energy consumption and save it in output_test
    @pyRAPL.measureit(output=output_test)
    def run_test():
        # subprocess.run(["python", os.path.join(os.getcwd(), f"test.py {path_to_checkpoint_file}")])
        main_test(path_to_checkpoint_file)

    for idx in range(n_measurements + 1):
        print(f"Script energy measurement #{idx}")
        run_test()

    test_df = output_test.data  # exctract df from measurement

    # first measurement has overhead, remove it
    n_sockets = len(test_df["socket"].unique())
    test_df = test_df.iloc[n_sockets:].reset_index(drop=True)

    # mean execution time of run_test() for base energy measurement
    mean_time_seconds = test_df["duration"].mean() / 1e6

    # Measure base energy consumption and save it in output_base
    @pyRAPL.measureit(output=output_base)
    def run_baseline(sleep_time):
        time.sleep(sleep_time)

    for idx in range(n_measurements):
        print(f"Baseline energy measurement #{idx+1}")
        run_baseline(mean_time_seconds)  # we measure the base energy consumed by the computer

    baseline_df = output_base.data

    # Save dataframes as csv
    test_df.to_csv("results/model_energy.csv", index=False, mode="w")
    baseline_df.to_csv("results/baseline_energy.csv", index=False, mode="w")

    # real energy is the difference between all energy and base energy
    real_energy_df = test_df.copy()
    real_energy_df[["pkg", "dram"]] = test_df[["pkg", "dram"]] - baseline_df[["pkg", "dram"]]

    # Compute values
    mean_energy_base_joules = n_sockets * (baseline_df["pkg"].mean() + baseline_df["dram"].mean()) / 1e6
    mean_energy_test_joules = n_sockets * (test_df["pkg"].mean() + test_df["dram"].mean()) / 1e6
    mean_energy_joules = n_sockets * (real_energy_df["pkg"].mean() + real_energy_df["dram"].mean()) / 1e6
    print(f"Base energy consumption = {mean_energy_base_joules} Joules")
    print(f"Solution energy consumption = {mean_energy_test_joules} Joules")
    print(f"Energy consumed by solution alone = {mean_energy_joules} Joules")

    return mean_energy_joules, mean_energy_base_joules


def modifiedF1score(csv_true_file="data/intrusion_big_train.csv", csv_pred_file="data/intrusion_big_train_pred.csv", l=300):
    # load data
    data_pred = pd.read_csv(csv_pred_file)
    data_true = pd.read_csv(csv_true_file).replace(
        to_replace=["sfa", "sha", "sya", "vna", "dfa"], value="malicious"
    )  # file to test data with labels will not be available to contestants

    # obtain labels
    labels_true = data_true["label"]
    labels_pred = pd.concat([data_pred["label"], pd.Series(["benign"] * (len(labels_true) - len(data_pred["label"])))], ignore_index=True)

    # errors = false_negatives + false_positives
    errors = (labels_pred != labels_true).sum()

    # get true positives
    true_positives = np.logical_and(labels_pred == labels_true, labels_true == "malicious").astype(float)

    # get idxs of attacts starting - t_0 in equation
    attack_idxs = np.where(np.logical_and(labels_true == "malicious", np.roll(labels_true, 1) == "benign"))[0]

    # scale_array 1+e^-(dt/lambda)
    if np.size(attack_idxs) != 0:
        scale_array = np.zeros_like(labels_true)
        for i in range(len(attack_idxs) - 1):
            delta_t = np.arange(attack_idxs[i + 1] - attack_idxs[i])
            scale_array[attack_idxs[i] : attack_idxs[i + 1]] = 1 + np.exp(-delta_t / l)
        delta_t = np.arange(len(scale_array) - attack_idxs[-1])
        scale_array[attack_idxs[-1] :] = 1 + np.exp(-delta_t / l)

        # scale true_positives with early detection bonus
        true_positives_mod = (true_positives * scale_array).sum()
    else:
        true_positives_mod = 0

    # compute f1 score
    f1 = true_positives_mod / (true_positives_mod + errors + 1e-6)
    print(f"F1 score = {f1}")

    return f1


def main(path_to_checkpoint_file, path_to_labeled_data, path_to_pred_data):
    mean_energy_joules, mean_energy_base_joules = energy_measurement(path_to_checkpoint_file)
    f1 = modifiedF1score(path_to_labeled_data, path_to_pred_data)

    print(f"F1 score: {f1} - Energy consumed: {mean_energy_joules} Joules")


if __name__ == "__main__":
    path_to_checkpoint_file = sys.argv[1]
    path_to_labeled_data = sys.argv[2]
    path_to_pred_data = sys.argv[3]

    # We deactivate GPU since energy measurement is only done in CPU
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    main(path_to_checkpoint_file, path_to_labeled_data, path_to_pred_data)

    # Restore the original value of CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
