# Cyber Threats Detection through Network Energy Consumption



## Description
Dataset, baseline example and evaluation code for the Challenge.

## Dataset
Train dataset can be found at `data/intrusion_big_train.csv`. Test dataset without labels can be found at `data/intrusion_test.csv`

## Setting up environment
```bash
git clone https://gitlab.kuleuven.be/networked-systems/public/cyber-threats-detection-through-network-energy-consumption-baseline.git
cd cyber-threats-detection-through-network-energy-consumption-baseline
conda env create -f env.yaml
conda activate challenge
```

## Training Baseline
We provide code where a Baseline model is trained using the train data in `data/intrusion_big_train.csv`
```bash
python main.py data/intrusion_big_train.csv
```

## Evaluating model 
For model evaluation we need to indicate the checkpoint path, the labeled data path and the predicted data path, using the pretrained model and the train dataset the evaluation can be executed by
```bash
python evaluation.py pretrained/checkpoint.ckpt data/intrusion_big_train.csv data/intrusion_big_train_pred.csv
```
## License

[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
