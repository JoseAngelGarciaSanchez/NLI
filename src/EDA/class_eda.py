import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
nltk.download('punkt')


class EDA_SNLI():

    def __init__(self, train, test, validation):
        self.train = pd.DataFrame(train)
        self.test = pd.DataFrame(test)
        self.validation = pd.DataFrame(validation) 
    
    def calculate_text_stats(self, data):
        data['text'] = data['premise'] + ' ' + data['hypothesis']
        # tokenize the text into individual words
        tokens = [word.lower() for sentence in data['text']
                  for word in nltk.word_tokenize(sentence)]
        # calculate the number of unique vocabulary words
        vocab = set(tokens) 
        num_vocab = len(vocab)
        
        # calculate the average number of words per sentence
        num_sentences = len(data)
        num_words = len(tokens)
        avg_words_per_sentence = num_words / num_sentences

        print('Number of unique vocabulary words in  :', num_vocab)
        print('Average number of words per sentence in :', avg_words_per_sentence)

        return num_vocab, avg_words_per_sentence


    def map_labels(self, data):
        # create a dictionary to map labels to output format
        label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction', -1: '-'}

        # map the labels column using the label_map dictionary
        data['labels'] = data['label'].map(label_map)

        # drop the original label column
        data = data.drop('label', axis=1)

        return data


    def plot_label_distribution(self, data, title):
        label_counts = data['labels'].value_counts()

        # bar plot for labels distribution
        sns.set_style("white")
        sns.barplot(x=label_counts.index, y=label_counts.values, palette='Blues')

        # add frequency labels above each bar
        for i, v in enumerate(label_counts.values):
            plt.text(i, v + 50, str(v), ha='center', fontsize=12)

        # set the title and axis labels
        plt.title(title, fontsize=14)
        plt.xlabel('Label', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        plt.show()


    def word_count(self, dataset, column):
        len_vector = []
        for text in dataset[column]:
            len_vector.append(len(text.split()))

        return len_vector


    def plot_length_distribution(self, data_premise, data_hypothesis, title):
        # creation of custom palette
        custom_palette = {'Premise': 'red', 'Hypothesis': 'blue'}

        # create a grid of subplots with 2 rows and 1 column
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

        # add the second subplot for the box plot
        ax1 = plt.subplot(gs[0])
        sns.boxplot(data=pd.DataFrame(
            {'Premise': data_premise, 'Hypothesis': data_hypothesis}), ax=ax1, orient='h', palette=custom_palette)
        ax1.set_xlabel('Sentence Type', fontsize=12)
        # ax1.set_ylabel('Word Count', fontsize=12)
        ax1.set_title(title, fontsize=14)

         # add the first subplot for the distribution plot
        ax0 = plt.subplot(gs[1])
        sns.distplot(data_premise, ax=ax0, label='Premise', color=custom_palette['Premise'])
        sns.distplot(data_hypothesis, ax=ax0, label='Hypothesis', color=custom_palette['Hypothesis'])
        #ax0.set_title('Distribution of Word Count in Train Dataset', fontsize=14)
        ax0.set_xlabel('Sentence length', fontsize=12)
        ax0.set_ylabel('Density', fontsize=12)
        ax0.legend()

        # adjust the space between subplots
        plt.subplots_adjust(hspace=0.3) 

        plt.show()