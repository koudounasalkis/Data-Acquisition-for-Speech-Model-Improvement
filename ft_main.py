import torch
import pandas as pd
import argparse
import numpy as np
import os
import math
import random
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
        "--epochs",
        help="number of training epochs",
        default=5,      # 5 for FSC, 7 for ITALIC
        type=int,
        required=False)
    parser.add_argument(
        "--steps",
        help="number of steps per epoch",
        default=500,    # 500 for FSC, 850 for ITALIC
        type=int,
        required=False)
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,      # 1 for FSC, 4 for ITALIC
        type=int,
        required=False)
    parser.add_argument(
        "--warmup_steps",
        help="number of warmup steps",
        default=500,    # 500 for FSC, 5000 for ITALIC
        type=int,
        required=False)
    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
        required=False)
    parser.add_argument(
        "--model",
        help="model to use for training, \
            either 'facebook/wav2vec2-base' for FSC \
            or 'facebook/wav2vec2-xls-r-300m' for ITALIC",
        default="facebook/wav2vec2-xls-r-300m",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_folder",
        help="path to the df folder",
        default="data/dataset-name",
        type=str,
        required=False) 
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="dataset_model-name",
        type=str,
        required=False)
    parser.add_argument(
        "--dataset",
        help="name of the dataset, either 'fsc' or 'italic'",
        default="fsc",
        type=str,
        required=False)
    parser.add_argument(
        "--approach",
        help="approach to be used for bias mitigation: \
            one of 'clustering', 'divexplorer', 'random'",
        default="divexplorer",
        type=str,
        required=False)
    parser.add_argument(
        "--balancing",
        help="whether to use weights rebalancing",
        action="store_true",
        required=False)
    parser.add_argument(
        "--verbose",
        help="whether to print logging information",
        action="store_true",
        required=False)
    parser.add_argument(
        "--num_clusters",
        help="Number of clusters to be used for clustering approach",
        default=10,
        type=int,
        required=False)
    parser.add_argument(
        "--num_problematic_subgroups",
        help="Number of problematic subgroups to be used for mitigation",
        default=2,
        type=int,
        required=False)
    parser.add_argument(
        "--num_samples",
        help="Number of samples to be used for random approach",
        default=226,
        type=int,
        required=False)
    parser.add_argument(
        "--seed",
        help="Seed to be used for reproducibility",
        default=42,
        type=int,
        required=False)
    args = parser.parse_args()
    return args



""" Read and Process Data"""
def read_data(df_folder, dataset, balancing, k=2, num_samples=226, approach='divexplorer', verbose=False):
    if balancing:
        df_train = pd.read_csv(
            os.path.join(df_folder, 'new_data', f'train_data_{approach}_k{k}_{num_samples}.csv'), 
            index_col=None)
        df_valid = pd.read_csv(
            os.path.join(df_folder, 'valid_data.csv'), 
            index_col=None)
    else:
        df_train = pd.read_csv(
            os.path.join(df_folder, 'train_data_80.csv'), 
            index_col=None)
        df_valid = pd.read_csv(
            os.path.join(df_folder, 'valid_data.csv'), 
            index_col=None)
    if verbose:
        print("Train size: ", len(df_train))
        print("Valid size: ", len(df_valid))

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
    for index in range(0,len(df_valid)):
        df_valid.loc[index,'label'] = label2id[df_valid.loc[index,'intent']]
    df_valid['label'] = df_valid['label'].astype(int)

    return df_train, df_valid, num_labels, label2id, id2label
 

""" Main Program """
if __name__ == '__main__':

    ## Parse command line parameters
    args = parse_cmd_line_params()
    num_epochs = args.epochs
    num_steps = args.steps
    num_wu = args.warmup_steps
    num_gas = args.gradient_accumulation_steps
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)    

    if args.verbose:
        print("------------------------------------")
        print("Command line parameters:")
        print(f"Batch size: {args.batch}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Number of steps per epoch: {num_steps}")
        print(f"Learning rate: {args.lr}")
        print(f"Model: {args.model}")
        print(f"Dataset folder: {args.df_folder}")
        print(f"Output directory: {output_dir}")
        print(f"Dataset: {args.dataset}")
        print(f"Balancing: {args.balancing}")
        if args.balancing:
            print(f"Approach: {args.approach}")
            print(f"Number of problematic subgroups: {args.num_problematic_subgroups}")
        print("------------------------------------\n")


    ## Set seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ## Model & Feature Extractor
    model_checkpoint = args.model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    first_run = True

    for epoch in range(num_epochs):

        ## Train & Valid df
        df_train, df_valid, num_labels, label2id, id2label = read_data(
            df_folder=args.df_folder, 
            dataset=args.dataset,
            balancing=args.balancing, 
            k=args.num_problematic_subgroups,
            num_samples=args.num_samples,
            approach=args.approach, 
            verbose=args.verbose
            )

        ## Loading Model
        if not first_run:
            steps = epoch * num_steps
            model_checkpoint = os.path.join(output_dir, f"checkpoint-{steps}")
        if args.verbose:
            print("------------------------------------")
            print(f"Loading model from {model_checkpoint}")
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            local_files_only=not first_run
            )
        if args.verbose:
            print("Model loaded successfully!")
            print("------------------------------------\n")

        ## Train & Valid Datasets 
        if args.verbose:
            print("----------------------------------")
            print("Loading dataset...")
        max_duration = 4.0 # 10.0 for ITALIC, 4.0 for FSC
        train_dataset = Dataset(
            examples=df_train, 
            feature_extractor=feature_extractor, 
            max_duration=max_duration, 
            )
        valid_dataset = Dataset(
            examples=df_valid, 
            feature_extractor=feature_extractor, 
            max_duration=max_duration, 
            )
        if args.verbose:
            print("Dataset loaded successfully!")
            print("----------------------------------\n")

        ## Training Arguments
        if first_run:
            training_arguments = define_training_args(
                output_dir=output_dir, 
                batch_size=args.batch, 
                num_steps=num_steps, 
                lr=args.lr, 
                gradient_accumulation_steps=num_gas,
                warmup_steps=num_wu
                )
        else:
            training_arguments = define_training_args(
                output_dir=output_dir, 
                batch_size=args.batch, 
                num_steps=num_steps*(epoch+1), 
                lr=args.lr, 
                gradient_accumulation_steps=num_gas,
                warmup_steps=num_wu
                )

        ## Trainer 
        trainer = WeightedTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            balancing=args.balancing
            )

        ## Train and Evaluate
        if args.verbose:
            print("------------------------------------")
            print(f"Training the model at epoch {epoch+1}...")
            if args.balancing:
                print("Balancing option: activated!")
                if args.approach == 'clustering':
                    print("Approach: clustering")
                elif args.approach == 'divexplorer':
                    print("Approach: DivExplorer")
                elif args.approach == "random":
                    print("Approach: random")
            else:
                print("Balancing option: deactivated!")
            print("------------------------------------\n")

        if first_run:
            trainer.train()
            first_run = False
        else:
            trainer.train(resume_from_checkpoint=model_checkpoint)

        predictions = trainer.predict(valid_dataset).predictions
        df_valid['prediction'] = np.argmax(predictions, axis=1) == df_valid['label']
        df_valid['prediction'] = df_valid['prediction'].astype(int)
        df_valid.to_csv(os.path.join(output_dir, f"predictions_{epoch+1}.csv"), index=False)

    if args.verbose:
        print("Training completed successfully!")
        print("------------------------------------\n")