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
unique_set= []
dict = {}
count=0
fall =0
threshold=0.6
lines = lines[:10000]
with open('relation_unique.txt', 'r') as file:
    line = file.readlines()
    for item in line:
        unique_set.append(item.strip())
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
        if subject == S[j]:        # subject mathced
            if i==j:             # matched itself
                count +=1
            else:               # matched others (not uniquely matched, need retrieval)
                fall += 1
                ls.append(j)
                dict.update({i:ls})

ns_dict = {}
n_list_dict={}
ns_count=0
ns_fall =0
for i,line in enumerate(lines):
    neighbors = line['neighborhood_prompts']
    # subject = line['requested_rewrite']['subject']
    ls = []
    nls = []

    for nn,neighbor in enumerate(neighbors):
        searched = 0
        for j,sub in enumerate(S):
            if searched == 1:
                break
            if sub+' ' in neighbor:
                for rel in unique_set:            #  subject matched
                    if neighbor.replace(sub,'*') == rel:
                        print(sub, i)
                        ns_fall += 1
                        ls.append(j)
                        ns_dict.update({i:ls})
                        nls.append(nn)
                        n_list_dict.update({i: nls})
                        searched = 1
                        break

print(n_list_dict,len(n_list_dict))
print(ns_dict,len(ns_dict))
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
print("mdoel_sentenceTransformer")


#####################################   NS
ns_score=[]
ns_ret = {}
for item in ns_dict:      #  for multi-matched facts, retrieve top fact
    facts = []
    line = lines[item]
    paraphrases = line['paraphrase_prompts']
    neighbors = []
    for ii in n_list_dict[item]:
        neighbors.append(line['neighborhood_prompts'][ii])
    for ff in ns_dict[item]:
        ll = lines[ff]
        new_fact = ll['requested_rewrite']['prompt'].format(ll['requested_rewrite']['subject']) + ' ' + \
                   ll['requested_rewrite']['target_new']['str']

        facts.append(new_fact)
    n_embeddings = model.encode(neighbors)
    f_embeddings = model.encode(facts)
    corpus_embeddings = torch.tensor(f_embeddings)
    corpus_embeddings = corpus_embeddings.to('cuda')
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    query_embeddings = torch.tensor(n_embeddings)
    query_embeddings = query_embeddings.to('cuda')
    query_embeddings = util.normalize_embeddings(query_embeddings)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score,
                                top_k=1)


    ret = []
    for i in range(len(hits)):
        ns_score.append(float(hits[i][0]['score']))
        if hits[i][0]['score'] >= threshold:
            idx = hits[i][0]['corpus_id']
        else:
            idx = -1
        ret.append(ns_dict[item][idx])
        ns_ret.update({item:ret})

ns_all_facts = {}
for ns_idx in n_list_dict:
    fc_list = []
    fc_idx=0
    for i in range(10):   # 10 neighbors for each sample
        if i in n_list_dict[ns_idx]:    # retrieved facts
            fc_list.append(ns_ret[ns_idx][fc_idx])
            fc_idx +=1
        else:
            fc_list.append(-1)
    ns_all_facts.update({ns_idx:fc_list})

for i in range(len(lines)):   # not matched/retrieved samples
    if i not in n_list_dict:
        ns_all_facts.update({i: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]})
with open("ns_retrieved_facts.json", 'w') as jf:
    json.dump(ns_all_facts,jf)



#########################  PS
c1=0
c2=0
ret_dict={}
score_list = []
for item in dict:      #  for multi-matched facts, retrieve top fact
    # print(item)

    facts = []
    line = lines[item]
    paraphrases = line['paraphrase_prompts']
    neighbors = line['neighborhood_prompts']
    p1 = paraphrases[0]
    p2 = paraphrases[1]
    for ff in dict[item]:
        ll = lines[ff]
        new_fact = ll['requested_rewrite']['prompt'].format(ll['requested_rewrite']['subject']) + ' ' + \
                   ll['requested_rewrite']['target_new']['str']
        facts.append(new_fact)
    p_embeddings = model.encode(paraphrases)
    f_embeddings = model.encode(facts)
    corpus_embeddings = torch.tensor(f_embeddings)
    corpus_embeddings = corpus_embeddings.to('cuda')
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    query_embeddings = torch.tensor(p_embeddings)
    query_embeddings = query_embeddings.to('cuda')
    query_embeddings = util.normalize_embeddings(query_embeddings)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score,
                                top_k=1)

    score_list.append(float(hits[0][0]['score']))
    score_list.append(float(hits[1][0]['score']))
    if hits[0][0]['score'] >= threshold:
        idx1 = hits[0][0]['corpus_id']
    else:
        idx1 = -1
    if hits[1][0]['score'] >= threshold:
        idx2 = hits[1][0]['corpus_id']
    else:
        idx2 = -1
    ret = [dict[item][idx1],dict[item][idx2]]
    ret_dict.update({item:ret})

all_facts = ret_dict
for i in range(len(lines)):
    if i not in ret_dict:
        all_facts.update({i:[i,i]})       # for * uniquely matched itself

with open("retrieved_facts.json", 'w') as jf:
    json.dump(all_facts,jf)

