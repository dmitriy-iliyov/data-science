import pandas as pd

import text_pre_processing as tpp
import bags
import task_2 as t2
import task_3 as t3


with open('files/doc5.txt', 'r') as file:
    lines = []
    for line in file:
        lines.append(line.strip())
corpus = list(filter(None, lines))

# prepare
prepared_corpus = tpp.end_preprocessing(corpus)
print(prepared_corpus)

# 1
df = bags.bag_of_bigrams(prepared_corpus)
df.to_csv('files/ngrams.csv')
print("Vector for 'Roman architecture':")
print(bags.bigram_phrase_vector(prepared_corpus, "Roman architecture"))

# 2
df1 = t2.to_tf_idf(prepared_corpus)
clusters = t2.clusterizate(prepared_corpus)
cluster_labels = pd.DataFrame(clusters, columns=['cluster'])
df1_with_clusters = pd.concat([df1, cluster_labels], axis=1)
print(df1_with_clusters)

# 3
w2v_model = t3.to_Word2Vec(prepared_corpus)
t3.task_3(w2v_model)
