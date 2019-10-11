# DPMI Codes

The command line is not implemented yet. Please read through all comments in the scripts and modify parameters manually.

## Requirements
1.  Make sure you have python 3 installed. (Tested for 3.5.4 and 3.6.5)
2.  Install dependencies with the following command:
```bash
pip install numpy matplotlib seaborn sklearn scipy xlrd pandas tensorflow-gpu keras shap
```

## Quick Demo
Some sample data are prepared to illustrate the experiment result:
1. Directly run `python analyze.py` in bash/command prompt to show 4 figures for classification performance.
2. Directly run `python ranking.py` in bash/command prompt to show importance distribution for the sample data `sample_data/test_1_0.xls`. A picture `test_1_0_importance.png` and a feature importance list `test_1_0_ranking.csv` will also be generated when the script ends.


## Prepare Data
### Feature List
The feature list of each sample should be a .xls file structured as follow:

| m/z | t | I | Z |
|--------|--------|--------|--------|
| xxx.xx |xxx.xx|xxxx|x|

m/z: mass to charge, t: retention time, I: intensity/abundance, Z: charge (optional, not used)
See `sample_data/test_1_0.xls` for reference.
### File Hierarchy
Feature lists of samples which belong to the same group should be put into the same folder:
```
root_dir
├─ group_A
│    ├ feature_list_a1.xls
│    ├ feature_list_a2.xls
│    └ ...
├─ group_B
│    ├ feature_list_b1.xls
│    └ ...
└ ...
```
### Convert to TFRecord
To apply CNN, all feature lists have to be converted to a image like matrix.

The `convert_and_cross_split_xls` function from `data2record.py` will split the data into multiple folds, convert them to matrix and write them into .tfrecord files. You can call it in your script, or modify the parameters in `'__main__'` and run the .py file in bash.

For each fold, a train data record, a test data record, a train data meta file and a test data meta file will be generated.

## Sample Classification
### Train
Modify parameters in `train.py` and run `python train.py`.

The final model of each fold will be generated in the specified folder.

### Test Analysis
The test accuracy should come along with the training log. But with the `test.py`, a statistics .csv file could be generated for further visualization. Run `python analyze.py` with its path configured and you could see the performance of the model.

## Importance Analysis with SHAP
### Generate SHAP Distribution
Modify `shap_test.py` and run it in bash will generate an importance distribution for specified sample.
### Rank Features
Modify `ranking.py` and run it in bash will calculate all feature's importance for specified sample.
