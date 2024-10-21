# https://nbviewer.org/github/MuhammadMoinFaisal/LargeLanguageModelsProjects/blob/main/Fine-Tune-Llama2/Fine_Tune_Llama2.ipynb

import os
import torch
import json
import gc 
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
model_names = [
            'meta-llama/Llama-2-7b-chat-hf',
            'mosaicml/mpt-7b-instruct',
            'tiiuae/falcon-7b',
            'meta-llama/Llama-2-13b-chat-hf'
            ]

# dataset_names = ['ambig_qa', 'disent_qa', 'hotpotqa']
# dataset_names = ['disent_qa']
dataset_names = ['distractor_hotpotqa']

# generation parameter
batch_size = 2
shots = ['1']

conflicting = True

# load model
for model_name in model_names:
    print(f'model_name: {model_name}')
    
    for dataset_name in dataset_names:
        saving_model_name = f'{model_name.split("/")[-1]}_{dataset_name}'
        # Merge the Base Model with the Trained Adapter
        # Reload model in FP16 and merge it with LoRA weights
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            load_in_8bit=True, # for llama13B
            device_map={"": 0},
            # cache_dir='/rc_scratch/sash9910/' # blanca
            # cache_dir='/scratch/alpine/sash9910' # alpine
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        #Reload the Base Model and load the QLoRA adapters
        model = PeftModel.from_pretrained(model, f'finetuned_models/{saving_model_name}'.replace('distractor_', '').replace('paraphrased_',''))
        model = model.merge_and_unload()
        if 'hotpotqa' in dataset_name :
            max_new_tokens = 100 # IF HOTPOTQA THIS NEEDS TO BE 100 BECAUSE THE ANSWER IS LONGER
        else:
            max_new_tokens = 40 

        context_length_dicts = {
            'llama':4096 - max_new_tokens,
            'mpt':2048 - max_new_tokens, # can handle 8k but runs out of memory so lowering this
            'falcon':2048 - max_new_tokens} 
        for model_key in context_length_dicts.keys():
            if model_key in model_name:
                model_max_length = context_length_dicts[model_key]

        print(f'dataset_name: {dataset_name}')
        # load dataset
        if dataset_name == 'disent_qa': # disent_qa
            with open('datasets/disent_qa_train.json', 'r') as fp:
                data_train = json.load(fp)
            with open('datasets/disent_qa_test.json', 'r') as fp:
                data_test = json.load(fp)
            if not conflicting: # for disentqa we also have a non conflicting setting
                test_instruction = [f'Question: {question}. Context: {original_context}. Answer: ' for question, original_context in zip(data_test['question'], data_test['original_context'])]
                instruct_test_dataset = {'instruction':test_instruction}
                instruct_test_dataset = Dataset.from_dict(instruct_test_dataset) # convert to HF dataset                       
            
            else:
                test_instruction = [f'Question: {question}. Context: {context}. Answer: ' for question, context in zip(data_test['question'], data_test['cited_context'])]
                instruct_test_dataset = {'instruction':test_instruction}
                instruct_test_dataset = Dataset.from_dict(instruct_test_dataset) # convert to HF dataset

        elif dataset_name == 'ambig_qa': # ambig_qa
            with open('datasets/ambig_qa_train.json', 'r') as fp:
                data_train = json.load(fp)
            with open('datasets/ambig_qa_test.json', 'r') as fp:
                data_test = json.load(fp)
            test_instruction = [f'Question: {question}. Context: {context}. Answer: ' for question, context in zip(data_test['question'], data_test['cited_context'])]
            
            instruct_test_dataset = {'instruction':test_instruction}
            instruct_test_dataset = Dataset.from_dict(instruct_test_dataset) # convert to HF dataset

        elif dataset_name == 'hotpotqa' or dataset_name == 'distractor_hotpotqa': # hotpotqa
            with open('datasets/hotpotqa_train_final.json', 'r') as fp:
                data_train = json.load(fp)
            with open('datasets/hotpotqa_test_final.json', 'r') as fp:
                data_test = json.load(fp)      
            
            if dataset_name != 'distractor_hotpotqa': # no distractors
                test_instruction = [f'Question: {question}. Context: {oracle_context} {new_conflicting_contexts_1} {new_conflicting_contexts_2}.\
                    Answer: '\
                    for question, oracle_context, new_conflicting_contexts_1, new_conflicting_contexts_2\
                    in zip(data_test['question'], data_test['cited_oracle_contexts'], data_test['new_conflicting_contexts_1'],data_test['new_conflicting_contexts_2'])]
            else:
                test_instruction = [f'Question: {question}. Context: {cited_context}. Answer: '\
                    for question, cited_context\
                    in zip(data_test['question'], data_test['cited_context'])]

            # hotpotqa_labels = [f'{cited_original_answers}. {new_cited_conflicting_answers_1}. {new_cited_conflicting_answers_2}' for cited_original_answers, new_cited_conflicting_answers_1, new_cited_conflicting_answers_2\
            # in zip(data_test['cited_original_answers'], data_test['new_cited_conflicting_answers_1'], data_test['new_cited_conflicting_answers_2'])]
            # instruct_test_dataset = {'instruction':test_instruction, 'label':hotpotqa_labels}
            instruct_test_dataset = {'instruction':test_instruction}
            instruct_test_dataset = Dataset.from_dict(instruct_test_dataset) # convert to HF dataset

        # choose shot
        for shot in shots:
            # print(f'shot: {shot}')
            gold_answers = []
            test_generated_answers = []
            prompts_list = []

            for b in range(0, len(data_test['question']), batch_size):
                if dataset_name != 'hotpotqa' and dataset_name != 'distractor_hotpotqa': # hotpot is different
                        gold_answers += [[orig_ans, conf_ans] for orig_ans,conf_ans in zip(data_test['original_answer'][b:b+batch_size], data_test['conflicting_answer'][b:b+batch_size])]
                
                else: # hotpot
                    gold_answers += [[orig_ans, conf_ans1, conf_ans2] for orig_ans,conf_ans1,conf_ans2 in
                                    zip(data_test['original_answer'][b:b+batch_size], data_test['conflicting_answers_1'][b:b+batch_size], data_test['conflicting_answers_2'][b:b+batch_size])]

                in_context_prompts = instruct_test_dataset[b:b+batch_size]['instruction']
                tokenized_input = tokenizer(instruct_test_dataset[b:b+batch_size]['instruction'], padding=True, return_tensors='pt', truncation=True, max_length=model_max_length)
                with torch.no_grad():
                    outputs = model.generate(input_ids=tokenized_input['input_ids'].to(device), attention_mask=tokenized_input['attention_mask'].to(device),
                                            do_sample=False, num_beams=2, early_stopping=True, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens)
                                            # pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens) # for 13b on hotpotqa     

                model_generated_answer = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                clean_answers = [model_generated_answer[i][len(tokenizer.decode((tokenized_input['input_ids'][i]),clean_up_tokenization_spaces=True, skip_special_tokens=True)):] for i in range(len(model_generated_answer))]
                test_generated_answers += clean_answers 
                prompts_list += in_context_prompts

            # save results
            if 'hotpotqa' in dataset_name: # hotpot doesnt need this
                results_dict = {'cited_original_answers': data_test['cited_original_answers'],
                            'cited_conflicting_answers_1': data_test['new_cited_conflicting_answers_1'], 
                            'cited_conflicting_answers_2': data_test['new_cited_conflicting_answers_2'],
                            'question': data_test['question'], 
                            'model_generated_answer': test_generated_answers, 'prompts_list':prompts_list,
                            'gold_answers':gold_answers}

            else:
                results_dict = {'original_answer': data_test['original_answer'],'conflicting_answer': data_test['conflicting_answer'], 'question': data_test['question'],
                    'cited_answer': data_test["cited_answer"], 'model_generated_answer': test_generated_answers, 'gold_answers':gold_answers, 'prompts_list':prompts_list}

            if not conflicting: # for disentqa we also have a non conflicting setting
                with open(f'results/finetune/NO_CONFLICT_{shot}_shot_results_{dataset_name}_{saving_model_name}.json', 'w') as fp:
                    json.dump(results_dict, fp)
            else:
                with open(f'results/finetune/{shot}_shot_results_{dataset_name}_{saving_model_name}.json', 'w') as fp:
                    json.dump(results_dict, fp)
                
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()