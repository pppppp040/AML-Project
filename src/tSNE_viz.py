import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import re
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

def parseFile(fileName):
    rawData = pd.read_csv(fileName, header = None)
    return rawData

def load_embeddings(embedding_file):
    word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary = True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))
    return word2vec

def perform_tSNE(unique_tags, word2vec):
    topic_order = []
    tag_word_vecs = []
    for topic, tags in unique_tags.iteritems():
        vecs = []
        for tag in tags:
            vecs.append(word2vec.word_vec(tag))

        tag_word_vecs.extend(vecs)

        topic_order.append(topic)

    unique_tags['topic_order'] = topic_order
    model = TSNE(n_components = 2, random_state = 0)
    # np.set_printoptions(suppress = True)
    tag_word_vecs_tsne = model.fit_transform(tag_word_vecs)

    return tag_word_vecs_tsne

def get_unique_tags_from_word2vec(lst_tags_strings, word2vec):

    unique_tags = []
    for tag_str in lst_tags_strings:
        tag_str = re.sub(r"[^A-Za-z0-9]", " ", tag_str)
        tag_str_words = tag_str.strip().split()
        unique_tags.extend(set(tag_str_words))

    unique_tags = set(unique_tags)

    new_unique_tags = []
    for tag in unique_tags:
        if tag in word2vec.vocab:
            new_unique_tags.append(tag)

    return new_unique_tags

def vizualize_tSNE(unique_tags, tag_word_vecs_tsne, markers, colors):
    vecs_processed = 0

    for ii, topic in enumerate(unique_tags['topic_order']):

        topic_vecs_len = len(unique_tags[topic])

        tag_word_vecs_tsne_x = tag_word_vecs_tsne[vecs_processed:vecs_processed + topic_vecs_len, 0]
        tag_word_vecs_tsne_y = tag_word_vecs_tsne[vecs_processed:vecs_processed + topic_vecs_len, 1]

        plt.scatter(tag_word_vecs_tsne_x,
                tag_word_vecs_tsne_y,
                marker = markers[ii],
                # the color
                color = colors[ii],
                # the alpha
                alpha = 0.7,
                # with size
                s = 124,
                # labelled this
                label = topic)

        vecs_processed += topic_vecs_len

    plt.title('tSNE visualization of tags in different domains')

    plt.ylabel('y')

    plt.xlabel('x')

    plt.legend(loc = 'upper right')

    plt.show()

def main():
    topic2FileNames = {'biology': 'biology_processed.csv', 'cooking': 'cooking_processed.csv', 'crypto': 'crypto_processed.csv', 'diy': 'diy_processed.csv', 'robotics': 'robotics_processed.csv', 'travel': 'travel_processed.csv'}
    markers = ['x', 'o', '^', '*', '+', '.']
    colors = rainbow(np.linspace(0, 1, len(topic2FileNames)))
    BASE_DIR = '../Data/'
    embedding_file = BASE_DIR + 'GoogleNews-vectors-negative300.bin'

    word2vec = load_embeddings(embedding_file)

    unique_tags = {}

    for topic, fn in topic2FileNames.iteritems():
        print 'Processing topic: ', topic
        rawData = parseFile('../ProcessedData/' + fn)
        utags = get_unique_tags_from_word2vec(rawData[3], word2vec)
        print 'Number of unique tags: ', len(utags)
        unique_tags[topic] = utags

    tag_word_vecs_tsne = perform_tSNE(unique_tags, word2vec)

    vizualize_tSNE(unique_tags, tag_word_vecs_tsne, markers, colors)

if __name__ == '__main__':
    main()
