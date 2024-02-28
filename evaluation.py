import os, subprocess, time, sys
import pyRAPL
import pandas as pd
import numpy as np


def energy_measurement(path_to_checkpoint_file):
    # Number of times test.py will be run to average the energy consumption
    n_measurements = 10

    # Setting up the pyRAPL library and choosing the measurements outputs
    pyRAPL.setup()
    output_test = pyRAPL.outputs.DataFrameOutput()  # save the outputs in a dataframe
    output_base = pyRAPL.outputs.DataFrameOutput()

    # Measure test.py energy consumption and save it in output_test
    @pyRAPL.measureit(output=output_test)
    def run_test():
        subprocess.run(["python", os.path.join(os.getcwd(), f"test.py {path_to_checkpoint_file}")])

    for _ in range(n_measurements + 1):
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

    for _ in range(n_measurements):
        run_baseline(mean_time_seconds)  # we measure the base energy consumed by the computer

    baseline_df = output_base.data

    # Save dataframes as csv
    test_df.to_csv("model_energy.csv", index=False, mode="w")
    baseline_df.to_csv("baseline_energy.csv", index=False, mode="w")

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
    data_true = pd.read_csv(csv_true_file)  # file to test data with labels will not be available to contestants

    # obtain labels
    labels_true = data_true["label"]
    labels_pred = data_pred["label"]

    # errors = false_negatives + false_positives
    errors = (labels_pred != labels_true).sum()

    # get true positives
    true_positives = np.logical_and(labels_pred == labels_true, labels_true == "malicious").astype(float)

    # get idxs of attacts starting - t_0 in equation
    attack_idxs = np.where(np.logical_and(labels_true == "malicious", np.roll(labels_true, 1) == "benign"))

    # scale_array 1+e^-(dt/lambda)
    scale_array = np.zeros_like(labels_true)
    for i in range(len(attack_idxs) - 1):
        delta_t = np.arange(attack_idxs[i + 1] - attack_idxs[i])
        scale_array[attack_idxs[i] : attack_idxs[i + 1]] = 1 + np.exp(-delta_t / l)
    delta_t = np.arange(len(scale_array) - attack_idxs[-1])
    scale_array[attack_idxs[-1] :] = 1 + np.exp(-delta_t / l)

    # scale true_positives with early detection bonus
    true_positives_mod = (true_positives * scale_array).sum()

    # compute f1 score
    f1 = true_positives_mod / (true_positives_mod + errors)

    return f1


def main(path_to_checkpoint_file, path_to_labeled_data, path_to_pred_data):
    mean_energy_joules, mean_energy_base_joules = energy_measurement(path_to_checkpoint_file)
    f1 = modifiedF1score(path_to_labeled_data, path_to_pred_data)

    print(f"F1 score: {f1} - Energy consumed: {mean_energy_joules} Joules")


if __name__ == "__main__":
    path_to_checkpoint_file = sys.argv[1]
    path_to_labeled_data = sys.argv[2]
    path_to_pred_data = sys.argv[3]
    main(path_to_checkpoint_file, path_to_labeled_data, path_to_pred_data)
