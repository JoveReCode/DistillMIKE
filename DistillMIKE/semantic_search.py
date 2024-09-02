from sentence_transformers import SentenceTransformer, util
import torch
import pickle
# with open('embeddings_retrieval_n.pkl', "rb") as fIn:
with open('embeddings_ori.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)

    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']
print(stored_embeddings.shape)
corpus_embeddings = torch.tensor(stored_embeddings[10000:])
query_embeddings = torch.tensor(stored_embeddings[:10000])
print(corpus_embeddings.shape)
print(query_embeddings.shape)
corpus_embeddings = corpus_embeddings.to('cuda')
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

query_embeddings = query_embeddings.to('cuda')
query_embeddings = util.normalize_embeddings(query_embeddings)

hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=60)

# print(hits)

for i, hit in enumerate(hits):
    # if i > 10:
    #     break
    # print("Query:", stored_sentences[i])
    # print("\n")
    # with open('corpus_idx_retrieval_score.txt', mode='a') as src:
    with open('corpus_idx_10k.txt', mode='a') as src:
        for k in range(len(hit)):
            # print(hit[k]['corpus_id']+2000*13, end=" ")
            h = hit[k]['score']
            # src.write(f'{h}')
            # src.write(str(hit[k]['corpus_id']+20000)+'\t'+str(hit[k]['score']))
            src.write(str(hit[k]['corpus_id']+10000+' '))
            # print(hit[k]['score'],stored_sentences[hit[k]['corpus_id']+10000])
        src.write('\n')
src.close()