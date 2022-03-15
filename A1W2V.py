import gensim.downloader as api

model = api.load('word2vec-google-news-300')

print(model.most_similar('Canada'))
print(model.most_similar(positive=['Canada', 'Russia'], negative=['Canadian']))
print(model.doesnt_match("breakfast cereal dinner lunch"))
print(model.similarity('woman', 'man'))
