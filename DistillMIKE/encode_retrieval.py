from sentence_transformers import SentenceTransformer
import pickle
import json

device = 'cuda'
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
print("mdoel_sentenceTransformer")
with open('./multi_counterfact.json', 'r') as f:
    lines = json.load(f)
print("datasize:",len(lines))
sentences = []
subjects = []
for i, line in enumerate(lines):

    new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject']) + ' ' + line['requested_rewrite']['target_new']['str']
    target_new = line['requested_rewrite']['target_new']['str']
    target_true = line['requested_rewrite']['target_true']['str']
    paraphrases = line['paraphrase_prompts']
    neighbors = line['neighborhood_prompts']
    subject = line['requested_rewrite']['subject']

    if i < 10000:
        for p in paraphrases:
            # sentences.append(f"New Fact: {new_fact}\nPrompt: {p}")
            sentences.append(f"{p}")
            subjects.append(subject)
    subjects.append(subject)
    # if i < 10000:
    #     for n in neighbors:
    #         # sentences.append(f"New Fact: {new_fact}\nPrompt: {p}")
    #         sentences.append(f"{n}")
    #         subjects.append(subject)
    # subjects.append(subject)
# for i, line in enumerate(lines):
#     if i < 10000:
#         new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject']) + ' ' + line['requested_rewrite']['target_new']['str']
#         new_fact_pre = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
#         subject = line['requested_rewrite']['subject']
#         sentences.append(f"{new_fact_pre}")
#         subjects.append(subject)

print("embedding start")
embeddings = model.encode(sentences)
print("embedding end")
print("writing to pkl")

#Store sentences & embeddings on disc
with open('embeddings_retrieval_para.pkl', "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings, 'subjects': subjects}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Load sentences & embeddings from disc
# with open('embeddings.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     stored_sentences = stored_data['sentences']
#     stored_embeddings = stored_data['embeddings']