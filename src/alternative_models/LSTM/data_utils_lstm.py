import numpy as np
import os
import wget
import urllib.request
import zipfile
import ssl



def download_data(glove:str='glove.6B.zip'):

    if not os.path.exists(f'./{glove}'):
        print(f'Downloading {glove}...')
        url = f'http://nlp.stanford.edu/data/{glove}'
        
        # create a custom opener to disable SSL verification
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
        
        # download the file using the custom opener
        response = opener.open(url)
        data = response.read()
        
        # save the downloaded data to file
        with open(glove, 'wb') as f:
            f.write(data)
        
        print(f'{glove} download complete!')
    else:
        print(f'{glove} already exists.')

    if not os.path.exists("./data"):
        os.mkdir("./data")

    print("Extracting...")
    with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')
    print("done!")

def load_glove_embeddings(path):
    embeddings_index = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def preprocess_dataset(dataset, embeddings):
    def preprocess_sentence(sentence):
        sentence = sentence.lower().split()
        sentence = [word.strip(",.") for word in sentence]
        return sentence

    def get_embedding(sentence, embeddings):
        embedding_dim = len(next(iter(embeddings.values())))
        sentence_embedding = np.zeros(embedding_dim)
        words = preprocess_sentence(sentence)
        word_count = 0
        for word in words:
            if word in embeddings:
                sentence_embedding += embeddings[word]
                word_count += 1
        if word_count > 0:
            sentence_embedding /= word_count
        return sentence_embedding

    premise_embeddings = []
    hypothesis_embeddings = []

    for index, row in dataset.iterrows():
        premise_embedding = get_embedding(row['premise'], embeddings)
        hypothesis_embedding = get_embedding(row['hypothesis'], embeddings)
        premise_embeddings.append(premise_embedding)
        hypothesis_embeddings.append(hypothesis_embedding)

    dataset['premise_embedding'] = premise_embeddings
    dataset['hypothesis_embedding'] = hypothesis_embeddings

    return dataset
