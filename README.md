# RELITC - Relevance-based Infilling for Textual Counterfactuals

Official repository containing the scripts to run the experiments of the paper "Relevance-based Infilling for Natural Language Counterfactuals".


## Setup

#### Environment

Install the environment with Anacoda:

```
conda env create -f environment.yml
```

#### Data

The subfolders in `data` contain instructions to download the datasets.


#### Models

The black-box classifiers used for the `Yelp` and `OLID` datasets are automatically downloaded from the Hugging Face Hub. The ones used for the `yelp_sentence` and `call_me` datasets are stored in the ZENODO repository of the project (TBA).

The fine-tuned Conditional Masked Language Models used in the paper are stored in the ZENODO repository as well (TBA).


## Run experiments

#### Extract feature importances

In this step, the feature importances are extracted from all the texts in the datasets. This can be done by running the following command:

```
bash scripts/extract_feature_importance.sh config/shared_configs.sh config/tasks/{dataset}/{dataset}-wrt_predicted.sh
```

where `dataset` is one of `{Yelp, OLID, yelp_sentence, call_me}`.

This step reads the input datasets, eventually splits them into train, validation and test set, and extracts the feature importances with respect to the class predicted by the corresponding black box.


#### Fine-tune the Conditional Masked Language Model

This step fine-tunes the pretrained Conditional Masked Language Model and store the model in the `results/finetuned_models/{dataset}/` folder. The command is:

```
bash scripts/finetune_conditional_masked_language_model.sh config/shared_configs.sh config/tasks/{dataset}/{dataset}-wrt_predicted.sh config/tasks/{dataset}/{dataset}-finetuning_config.sh
```


#### Generate counterfactuals

Once the Conditional Masked Language Model has been fine-tuned on the dataset, counterfactuals are generated through the following command:

```
bash scripts/generate_counterfactuals.sh config/shared_configs.sh config/tasks/{dataset}/{dataset}-wrt_predicted.sh config/tasks/{dataset}/{dataset}-generation_config.sh "{infilling_order}" "{direction}t"
```

where `{infilling_order}` can be chosen among `[ordered, confidence]`, and `{direction}` among `[left_to_right, right_to_left]` when `infilling_order=ordered`, and `{direction}` among `[highest_first, lowest_first]` when `infilling_order=confidence`.

The generated counterfactuals are saved in `results/generated_counterfactuals/{dataset}/`.


## Citation

```bibtex
@inproceedings{
  ...
}
```
