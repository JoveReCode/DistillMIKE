from sentence_transformers import SentenceTransformer
import pickle
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
device = 'cuda'
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
print("mdoel_sentenceTransformer")
with open('./zsre_eval.json', 'r') as ef:
    elines = json.load(ef)
print("eval datasize:",len(elines))
with open('./zsre_train.json', 'r') as tf:
    tlines = json.load(tf)
print("train datasize:",len(tlines))
sentences = []
subjects = []
eval_data = []
train_data = []


for i, line in enumerate(elines):
    # print(str(i) + "/" + str(len(lines)))
    new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject']) + ' ' + line['requested_rewrite']['target_new']['str']
    target_new = line['requested_rewrite']['target_new']['str']
    target_true = line['requested_rewrite']['target_true']['str']
    paraphrases = line['paraphrase_prompts']
    neighbors = line['neighborhood_prompts']
    subject = line['requested_rewrite']['subject']
    if i < 10000:
        # sentences.append(f"New Fact: {new_fact}\nPrompt: {line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])}")
        for para in paraphrases:
            sentences.append(f"New Fact: {new_fact}\nPrompt: {para}")
            subjects.append(subject)
        subjects.append(subject)
    else:
        sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}")
    subjects.append(subject)

print(len(subjects))
for i, line in enumerate(tlines):
    # print(str(i) + "/" + str(len(lines)))
    new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject']) + ' ' + line['requested_rewrite']['target_new']['str']
    target_new = line['requested_rewrite']['target_new']['str']
    target_true = line['requested_rewrite']['target_true']['str']
    paraphrases = line['paraphrase_prompts']
    neighbors = line['neighborhood_prompts']
    subject = line['requested_rewrite']['subject']

    sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}")
    subjects.append(subject)

print(len(subjects))

print("embedding start")
embeddings = model.encode(sentences)
print("embedding end")
print("writing to pkl")

#Store sentences & embeddings on disc
with open('embeddings_zsre.pkl', "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings, 'subjects': subjects}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Load sentences & embeddings from disc
# with open('embeddings.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     stored_sentences = stored_data['sentences']
#     stored_embeddings = stored_data['embeddings']