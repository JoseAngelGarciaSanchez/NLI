# Librairies import
import pandas as pd
import pandas as pd
from datasets import load_dataset
from transformers import BertModel, BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import spacy
from allennlp.predictors.predictor import Predictor
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
import torch
import itertools

def create_dataframes(dataset):
    """ Create Dataframes out of  SNLI dataset samples  """
    df_train = pd.DataFrame(dataset["train"])
    df_test = pd.DataFrame(dataset["test"])
    df_validation = pd.DataFrame(dataset["validation"])
    return df_train, df_test, df_validation

def drop_missing_labels(df):
    """ Function to drop the observations with missing label"""
    df = df.drop(df[df["label"] == -1].index)
    return df

def prepare_data(df):
    """ Function to put in list the components of SNLI dataset
        that will be used in the model """
    premises = df['premise'].tolist()
    hypotheses = df['hypothesis'].tolist()
    labels = df['label'].tolist()
    return premises, hypotheses, labels

# Semantic Role Labeller Implementation
def concatenate_sentences(premises, hypotheses):
    """ Concatenate the premises and hypothesis in a single input"""
    # Concatenate the input sentences
    return [premise + "\n" + hypothesis for premise, hypothesis in zip(premises, hypotheses)]


def split_chunks(sentence, max_chunk_size):
    """Split the sentence into chunks of maximum size"""
    return [sentence[i:i + max_chunk_size] for i in range(0, len(sentence), max_chunk_size)]


def process_chunks(chunks):
    """Process each chunk separately using spaCy"""""
    doc = None
    for chunk in chunks:
        if doc is None:
            doc = nlp(chunk)
        else:
            doc = nlp(chunk, disable=["parser", "ner"])
    return doc


def extract_srl_labels(doc):
    """Extract the verb-only SRL labels using AllenNLP"""
    srl_labels_chunk = predictor.predict(sentence=str(doc))
    return srl_labels_chunk


def get_srl_labels_list_tags(srl_labels):
    """
    Get a list of SRL tags from a list of SRL predictions.

    Args:
    srl_labels: list of SRL predictions, each containing a "verbs" list with "tags" for each verb

    Returns:
    List of SRL tags, with one tag per element
    """
    srl_labels_list_tags = []
    for elem in srl_labels:
        tags = [tag for verb in elem['verbs'] for tag in verb['tags']]
        srl_labels_list_tags.append(tags)

    return srl_labels_list_tags

