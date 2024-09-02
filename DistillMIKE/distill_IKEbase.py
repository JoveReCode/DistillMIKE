import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import set_seed
# from transformers import GPT2Tokenizer, OPTForCausalLM
import json
import argparse
import random
import pickle
import os
import os.path as osp
from peft import LoraConfig, TaskType, get_peft_model

# model_name = 'EleutherAI/gpt-j-6B'
model_name = "/raid2/qiaosb/memit/models/GPT-J_memit_10000_0"
dataset_name = './multi_counterfact_ori.json'
test_num = 10000
overflow = []

with open('corpus_idx_10k.txt', 'r') as fIn:
    lines = fIn.readlines()
    lines = [line[:-1] for line in lines]

    corpus_idx = [[int(idx) for idx in line.split()] for line in lines]

def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning for pretrained GPTs")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def construct_icl_examples(idx, demos):

    # order = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,  1, 1, 1, 1]
    # order = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    order = [1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1]
    random.shuffle(order)
    icl_examples = []
    demo_ids = corpus_idx[idx][:len(order)]

    for demo_id, o in zip(demo_ids, order):

        line = demos[demo_id - test_num]
        new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
        target_new = line['requested_rewrite']['target_new']['str']
        target_true = line['requested_rewrite']['target_true']['str']

        if o == 0:
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {new_fact} {target_new}\n\n')
        elif o == 1:
            prompt = random.choice(line['paraphrase_prompts'])
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_new}\n\n')

    icl_examples.reverse()
    return icl_examples


def step_eval(teacher, student, tokenizer, icl_examples, target, x, tag):
    # icl_examples=[]

    tgt_len = len(tokenizer.encode(' ' + target))
    max_len = 2047 - len(tokenizer.encode(f'{x} {target}'))

    # ICL
    encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
    prompt_encodings = tokenizer(' ' + f'{x} {target}', return_tensors='pt')
    student_ids = prompt_encodings['input_ids'].to(student.device)
    if tag == 'ns':    # GPT-J teacher for NS
        # print("ns")
        teacher_ids = prompt_encodings['input_ids'].to(teacher.device)
    else:       # MEMIT teacher for ES,PS
        if encodings['input_ids'].size(1) < 2048:
            teacher_ids = encodings['input_ids'].to(teacher.device)
        else:    # overflow
            icl_encodings = tokenizer(''.join(icl_examples), return_tensors='pt', max_length=max_len, truncation=True)
            en_codings = torch.cat([icl_encodings['input_ids'], prompt_encodings['input_ids']], dim=1)
            teacher_ids = en_codings.to(teacher.device)

    target_ids = teacher_ids.clone()
    target_ids[:, :-tgt_len] = -100

    student_target_ids = student_ids.clone()
    student_target_ids[:, :-tgt_len] = -100
    with torch.no_grad():
        teacher_outputs = teacher(teacher_ids, labels = target_ids)
        teacher_logits = teacher_outputs.logits

    student_outputs = student(student_ids, labels = student_target_ids)
    student_logits = student_outputs.logits
    student_loss = student_outputs.loss
    # print(teacher_logits.shape)
    # print(student_logits.shape)
    # s_len = student_logits.size(1)
    # t_len = teacher_logits.size(1)
    # if s_len < t_len:
    #     teacher_logits = teacher_logits[:,-s_len:,:]
    # else:
    #     student_logits = student_logits[:,-t_len:,:]
    teacher_logits = teacher_logits[:, -3:, :]
    student_logits = student_logits[:, -3:, :]
    # print(teacher_trunc_logits.shape)
    # print(student_loss)

    return teacher_logits, student_logits, student_loss



def distill(lines, teacher,  student, optimizer, T, soft_weight, label_weight):
    losses = [0.0, 0.0, 0.0]
    es_loss = 0.0
    ps_loss = 0.0
    ns_loss = 0.0

    kl_div_loss = torch.nn.KLDivLoss(log_target=True)
    example_idx = 0
    S = []
    for i, line in enumerate(lines):
        subject = line['requested_rewrite']['subject']
        S.append(subject)

    for i, line in enumerate(lines):
        if i % 10 == 0:
            print(f'{i} \t es_loss:{losses[0]} \t ps_loss:{losses[1]} \t ns_loss:{losses[2]}')

        prompt = line['requested_rewrite']['prompt']
        subject = line['requested_rewrite']['subject']
        prompt = prompt.format(subject)

        target_true = line['requested_rewrite']['target_true']['str']
        target_new = line['requested_rewrite']['target_new']['str']
        tragets = [target_new, target_true]

        # icl_examples = optimized_icl_examples(example_idx, demos)
        icl_examples = construct_icl_examples(example_idx, demos)

        icl_examples.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_new}\n\n')  # prompt
        example_idx += 1

        for target in tragets:
            # es_flag = 0
            # for sub in S:
            #     if (sub + ' ' in prompt) or (sub + '\'' in prompt) or (sub + ',' in prompt) or (sub + '.' in prompt):
            #         t_logits, s_logits, s_loss = step_eval(teacher, student, tokenizer, icl_examples, target,
            #                                                f'Prompt: {prompt}', 'es')
            #         es_flag = 1
            #         break
            # if es_flag == 0:
            #     t_logits, s_logits, s_loss = step_eval(teacher, student, tokenizer, [], target,
            #                                            f'{prompt}', 'es')

            t_logits, s_logits, s_loss = step_eval(teacher, student, tokenizer, icl_examples, target, f'Prompt: {prompt}', 'es')
            soft_target = torch.nn.functional.log_softmax( t_logits / T, dim=-1 )
            soft_prob = torch.nn.functional.log_softmax( s_logits / T, dim=-1)

            soft_target_clone = soft_target.clone().to(student.device)

            soft_target_loss = kl_div_loss(soft_prob, soft_target_clone)
            label_loss = s_loss
            # loss = soft_weight * soft_target_loss + label_weight * label_loss    #   with label_loss
            loss = soft_weight * soft_target_loss        # without label_loss
            # print(loss, soft_target_loss, label_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            es_loss += loss.item()
            losses[0] = es_loss / ((i+1)*2)


            paraphrases = line['paraphrase_prompts']
            for paraphrase in paraphrases:
                ps_flag = 0
                for sub in S:
                    if (sub + ' ' in paraphrase) or (sub + '\'' in paraphrase) or (sub + ',' in paraphrase) or (
                            sub + '.' in paraphrase):
                        t_logits, s_logits, s_loss = step_eval(teacher, student, tokenizer, icl_examples, target,
                                                               f'Prompt: {paraphrase}', 'ps')
                        ps_flag = 1
                        break
                if ps_flag == 0:
                    t_logits, s_logits, s_loss = step_eval(teacher, student, tokenizer, [], target,
                                                           f'{paraphrase}', 'ns')
                # t_logits, s_logits, s_loss = step_eval(teacher1, student, tokenizer, icl_examples, target, f'Prompt: {paraphrase}', 'ps')
                soft_target = torch.nn.functional.log_softmax(t_logits / T, dim=-1)
                soft_prob = torch.nn.functional.log_softmax(s_logits / T, dim=-1)

                soft_target_clone = soft_target.clone().to(student.device)

                soft_target_loss = kl_div_loss(soft_prob , soft_target_clone)
                label_loss = s_loss
                # loss = soft_weight * soft_target_loss + label_weight * label_loss  #   with label_loss
                loss = soft_weight * soft_target_loss  # without label_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ps_loss += loss.item()
            losses[1] = ps_loss / ((i+1)*2*2)

            # neighbors = line['neighborhood_prompts'][:2]
            neighbors = line['neighborhood_prompts']
            ns_count = 0
            for neighbor in neighbors:
                if ns_count >= 2:       # only distill 2 NS
                    break
                ns_flag = 0
                for sub in S:
                    if (sub + ' ' in neighbor) or (sub + '\'' in neighbor) or (sub + ',' in neighbor) or (
                            sub + '.' in neighbor):
                        t_logits, s_logits, s_loss = step_eval(teacher, student, tokenizer, icl_examples, target,
                                                               f'Prompt: {neighbor}', 'is')     #  maybe should not distill here
                        ns_flag = 1
                        break
                if ns_flag == 0:
                    ns_count += 1
                    t_logits, s_logits, s_loss = step_eval(teacher, student, tokenizer, [], target,
                                                           f'{neighbor}', 'ns')
                # t_logits, s_logits, s_loss = step_eval(teacher2, student, tokenizer, icl_examples, target, f'Prompt: {neighbor}', 'ns')
                soft_target = torch.nn.functional.log_softmax(t_logits / T, dim=-1)
                soft_prob = torch.nn.functional.log_softmax(s_logits / T, dim=-1)

                soft_target_clone = soft_target.clone().to(student.device)

                soft_target_loss = kl_div_loss(soft_prob, soft_target_clone)
                label_loss = s_loss
                # loss = soft_weight * soft_target_loss + label_weight * label_loss  #   with label_loss
                loss = soft_weight * soft_target_loss  # without label_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ns_loss += loss.item()
            # losses[2] = ns_loss / ((i + 1)*2*10)
            losses[2] = ns_loss / ((i + 1)*2*2)
    return losses



if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
    os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'


    args = parse_args()
    seed = args.seed
    set_seed(seed)
    print(torch.cuda.device_count())

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM)


    print("loading MEMIT teacher model ...")
    teacher = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to('cuda:0')
    # teacher = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda:0')
    # teacher = GPTJForCausalLM.from_pretrained(model_name).to('cuda')
    print("MEMIT teacher model loaded.")

    # print("loading GPT-J teacher model ...")
    # teacher2 = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B').to('cuda:2')
    # print("GPT-J teacher model loaded.")

    print("loading student model ...")
    student = GPTJForCausalLM.from_pretrained("memit/models/GPT-J_memit_10000_0").to('cuda:1')
    # student = GPTJForCausalLM.from_pretrained("./GPT-J_memit_10000_0").to('cuda:1')

    print("studnet model loaded.")
    print("set LoRA ...")
    student = get_peft_model(student, peft_config)
    # student = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda:1')
    # print(student)
    student.print_trainable_parameters()

    teacher.eval()


    print("loading tokenizer ...")
    tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-j-6B')
    print("tokenizer loaded.")
    print("set optimizer ...\n")
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    lines = []

    with open(dataset_name, 'r') as f:
        lines = json.load(f)
    icl_examples = []
    demos = lines[test_num:]
    print("demos:", len(demos))
    lines = lines[:test_num]
    print("lines:", len(lines))

    epoches = 10
    T = 1.0
    soft_weight = 1e5
    label_weight = 0.5

    for epoch in range(epoches):
        e_loss = distill(lines, teacher, student, optimizer, T=T, soft_weight=soft_weight, label_weight=label_weight)
        if not osp.exists(f"./distill_modelsIKEbase/{epoch}"):
            os.makedirs(f"./distill_modelsIKEbase/{epoch}")
        student.save_pretrained(f"distill_modelsIKEbase/{epoch}")
        print(f"Epoche Loss: {e_loss}\n")
        with open('Epoche_lossIKEbase.txt', mode='a') as src:
            src.write(f'epoch:{epoch} \t es_loss:{e_loss[0]} \t ps_loss:{e_loss[1]} \t ns_loss:{e_loss[2]}\n')







