import torch
import pandas as pd
import argparse
import numpy as np
import os
import math

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from dataset import Dataset
from ft_utils import WeightedTrainer, define_training_args, compute_metrics
    
import warnings
warnings.filterwarnings("ignore")

""" Define Command Line Parser """
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="batch_size")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=8, 
        type=int,
        required=False)
    parser.add_argument(
        "--feature_extractor",
        help="feature extractor used for training, \
            either 'facebook/wav2vec2-base' for FSC \
            or 'facebook/wav2vec2-xls-r-300m' for ITALIC",
        default="facebook/wav2vec2-base",  
        type=str,                          
        required=False)   
    parser.add_argument(
        "--model",
        help="model to evaluate",
        default="model/best_ckpt",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_folder",
        help="path to the df folder",
        default="data/fsc",
        type=str,
        required=False) 
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="results",
        type=str,
        required=False)
    parser.add_argument(
        "--dataset",
        help="name of the dataset",
        default="fsc",
        type=str,
        required=False)
    args = parser.parse_args()
    return args



""" Read and Process Data"""
def read_data(df_folder):
    df_train = pd.read_csv(os.path.join(df_folder, 'train_data_80.csv'), index_col=None)
    df_test = pd.read_csv(os.path.join(df_folder, 'test_data.csv'), index_col=None)
    print("Train size: ", len(df_train))
    print("Test size: ", len(df_test))

    ## Prepare Labels
    intents = df_train['intent'].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(intents):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Create label column
    for index in range(0,len(df_train)):
        df_train.loc[index,'label'] = label2id[df_train.loc[index,'intent']]
    df_train['label'] = df_train['label'].astype(int)

    ## Test
    for index in range(0,len(df_test)):
        df_test.loc[index,'label'] = label2id[df_test.loc[index,'intent']]
    df_test['label'] = df_test['label'].astype(int)

    return df_train, df_test, num_labels, label2id, id2label
 

""" Main Program """
if __name__ == '__main__':

    ## Fix seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Parse command line parameters
    args = parse_cmd_line_params()
    max_duration = 4.0  # 4.0 for FSC, 10.0 for ITALIC

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        
    ## Train & Test df
    df_train, df_test, num_labels, label2id, id2label = read_data(args.df_folder)

    ## Model & Feature Extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.feature_extractor)
    print("------------------------------------")
    print(f"Loading model from {args.model}")
    model = AutoModelForAudioClassification.from_pretrained(
        args.model, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        local_files_only=True
        )
    print("Model loaded successfully!")
    print("------------------------------------\n")

    ## Train & Test Datasets 
    print("----------------------------------")
    print("Loading FSC dataset...")
    test_dataset = Dataset(df_test, feature_extractor, max_duration)
    print("Dataset loaded successfully!")
    print("----------------------------------\n")

    ## Training Arguments
    training_arguments = define_training_args(args.output_dir, args.batch)

    ## Trainer 
    trainer = WeightedTrainer(
        model=model,
        args=training_arguments,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        balancing=False)

    ## Evaluate
    predictions = trainer.predict(test_dataset).predictions
    predictions = np.argmax(predictions, axis=1)
    df_test['prediction'] = np.where(df_test['label'] == predictions, 1, 0)
    df_test['prediction'] = df_test['prediction'].astype(int)
    df_test.to_csv(os.path.join(args.output_dir, 'df_test.csv'), index=False)