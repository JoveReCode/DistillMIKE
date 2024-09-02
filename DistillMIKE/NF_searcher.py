import pickle
import json
import torch
from sentence_transformers import SentenceTransformer, util

device = 'cuda'
with open('./multi_counterfact.json', 'r') as f:
    lines = json.load(f)
print("datasize:",len(lines))
sentences = []
subjects = []
NF= []
PS = []
NS = []
S = []
dict = {}
count=0
fall =0
lines = lines[:10000]
for i, line in enumerate(lines):

    new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject']) + ' ' + line['requested_rewrite']['target_new']['str']
    target_new = line['requested_rewrite']['target_new']['str']
    NF.append(new_fact)
    paraphrases = line['paraphrase_prompts']
    PS.append(paraphrases)
    neighbors = line['neighborhood_prompts']
    NS.append(neighbors)
    subject = line['requested_rewrite']['subject']
    S.append(subject)

for i,line in enumerate(lines):

    paraphrases = line['paraphrase_prompts']
    subject = line['requested_rewrite']['subject']
    ls = [i]
    for j,nf in enumerate(NF):
        # if subject+' ' in nf:
        if subject == S[j]:
            if i==j:
                count +=1
            else:
                # print(subject)
                # print(paraphrases)
                # print(nf)
                fall += 1
                ls.append(j)
                dict.update({i:ls})


# print(dict)
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
print("mdoel_sentenceTransformer")

c1=0
c2=0
for item in dict:

    facts = []
    line = lines[item]
    paraphrases = line['paraphrase_prompts']
    p1 = paraphrases[0]
    p2 = paraphrases[1]
    for ff in dict[item]:
        ll = lines[ff]
        new_fact = ll['requested_rewrite']['prompt'].format(ll['requested_rewrite']['subject']) + ' ' + \
                   ll['requested_rewrite']['target_new']['str']

        # new_fact = ll['requested_rewrite']['prompt'].format(ll['requested_rewrite']['subject'])
        facts.append(new_fact)

    # print("embedding start")
    p_embeddings = model.encode(paraphrases)
    f_embeddings = model.encode(facts)
    # print("embedding end")
    corpus_embeddings = torch.tensor(f_embeddings)
    corpus_embeddings = corpus_embeddings.to('cuda')
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    query_embeddings = torch.tensor(p_embeddings)
    query_embeddings = query_embeddings.to('cuda')
    # print(query_embeddings.shape)
    query_embeddings = util.normalize_embeddings(query_embeddings)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score,
                                top_k=1)

    # print(hits[0][0]['corpus_id'])
    if hits[0][0]['corpus_id'] != 0:
        c1+=1

        print(p1)
        print(facts[0])
        print(facts[hits[0][0]['corpus_id']])
    if hits[1][0]['corpus_id'] != 0:
        c2+=1
        print(p2)
        print(facts[0])
        print(facts[hits[1][0]['corpus_id']])
    # print(hits)



print(count)
print(fall)

print(c1)
print(c2)