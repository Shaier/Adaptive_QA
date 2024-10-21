# libraries
import transformers
from transformers import AutoTokenizer
import torch 
import json 
import os
import logging
import gc
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
model_names = [
            'meta-llama/Llama-2-7b-chat-hf',
            'mosaicml/mpt-7b-instruct',
            'tiiuae/falcon-7b',
            'meta-llama/Llama-2-13b-chat-hf'
            'meta-llama/llama-2-70b-chat-hf',
               ]

# dataset_names = ['ambig_qa', 'disent_qa']
# dataset_names = ['disent_qa']
# dataset_names = ['paraphrased_disent_qa']
# dataset_names = ['distractor_hotpotqa']
dataset_names = ['hotpotqa']

conflicting = True

# generation parameter
batch_size = 6 #torch.cuda.device_count() * 1 # each a100 can handle ~2

for dataset_name in dataset_names:
    if 'hotpotqa' in dataset_name :
        max_new_tokens = 100 # IF HOTPOTQA THIS NEEDS TO BE 100 BECAUSE THE ANSWER IS LONGER
    else:
        max_new_tokens = 40     
    context_length_dicts = {
        'llama':4096 - max_new_tokens,
        'mpt':2048 - max_new_tokens, # can handle 8k but runs out of memory so lowering this
        'falcon':2048 - max_new_tokens,
    }          
    # load model
    for model_name in model_names:
        for model_key in context_length_dicts.keys():
            if model_key in model_name:
                model_max_length = context_length_dicts[model_key]

        model_name_for_saving = model_name.split('/')[-1]

        # load dataset
        if dataset_name == 'disent_qa': # disent_qa
            with open('../datasets/disent_qa_train.json', 'r') as fp:
                data_train = json.load(fp)
            with open('../datasets/disent_qa_test.json', 'r') as fp:
                data_test = json.load(fp)
        elif dataset_name == 'paraphrased_disent_qa': # disent_qa
            with open('../datasets/disent_qa_train.json', 'r') as fp:
                data_train = json.load(fp) # load the regular train dataset
            with open('../datasets/paraphrased_disentqa_test.json', 'r') as fp:
                data_test = json.load(fp)                
        elif dataset_name == 'ambig_qa': # ambig_qa
            with open('../datasets/ambig_qa_train.json', 'r') as fp:
                data_train = json.load(fp)
            with open('../datasets/ambig_qa_test.json', 'r') as fp:
                data_test = json.load(fp)
        elif dataset_name == 'hotpotqa' or dataset_name == 'distractor_hotpotqa': # hotpotqa
            with open('../datasets/hotpotqa_train_final.json', 'r') as fp:
                data_train = json.load(fp)
            with open('../datasets/hotpotqa_test_final.json', 'r') as fp:
                data_test = json.load(fp)                

        # initialize config
        llm_config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # initialize model 
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,  
            device_map="auto",
            config=llm_config,
            torch_dtype=torch.bfloat16, # Load model weights in bfloat16
            load_in_8bit=True, # for llama13B
            trust_remote_code=True,
            )
        # initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation=True, model_max_length = model_max_length) # need to initialize it so I can't automatically find the model_max_length
        tokenizer.pad_token = tokenizer.eos_token # add padding token

        # loop through the data and generate
        gold_answers = []
        test_generated_answers = []
        prompts_list = []
        for b in range(0, len(data_test['question']), batch_size):
            if b % 10 == 0:
                print(f'index {b} out of {min(1000,len(data_test["question"]))}')
            # get a batch of questions
            batch_questions = data_test['question'][b:b+batch_size]
            # get a batch of gold answers (original and conflicting)
            if dataset_name != 'hotpotqa' and dataset_name != 'distractor_hotpotqa': # hotpot is different
                gold_answers += [[orig_ans, conf_ans] for orig_ans,conf_ans in zip(data_test['original_answer'][b:b+batch_size], data_test['conflicting_answer'][b:b+batch_size])]
            else: # hotpotqa
                gold_answers += [[orig_ans, conf_ans1, conf_ans2] for orig_ans,conf_ans1,conf_ans2 in
                                zip(data_test['original_answer'][b:b+batch_size], data_test['conflicting_answers_1'][b:b+batch_size], data_test['conflicting_answers_2'][b:b+batch_size])]
        
            # 1 shot prompt -- conflict setting
            if conflicting:
                if 'hotpotqa' in dataset_name:
                    if dataset_name == 'distractor_hotpotqa': # for the distractor hotpot we use the cited_context entry
                        in_context_prompts = [f'Question: {data_test["question"][b + test_entry]}. \
                                            Context: {data_test["cited_context"][b + test_entry]}. Answer:' for test_entry in range(len(batch_questions))]        
                    else: # for the non-distractor hotpot we use the oracle contexts entries
                        in_context_prompts = [f'Question: {data_test["question"][b + test_entry]}. Context: {data_test["cited_oracle_contexts"][b + test_entry]} {data_test["new_conflicting_contexts_1"][b + test_entry]} {data_test["new_conflicting_contexts_2"][b + test_entry]}. Answer:' for test_entry in range(len(batch_questions))]
                else:
                    in_context_prompts = [f'Question: {data_test["question"][b + test_entry]}. \
                                        Context: {data_test["cited_context"][b + test_entry]}. Answer:' for test_entry in range(len(batch_questions))]
                
            else:
                # 1 shot prompt -- original context only (no conflict)
                in_context_prompts = [f'Question: {data_test["question"][b + test_entry]}. \
                                    Context: {data_test["original_context"][b + test_entry]}. Answer:' for test_entry in range(len(batch_questions))]                        
            
            # tokenize the input prompt
            tokenized_input = tokenizer(in_context_prompts, padding=True, return_tensors='pt', truncation=True)
            # generate
            with torch.no_grad():
                outputs = model.generate(input_ids=tokenized_input['input_ids'].to(device), attention_mask=tokenized_input['attention_mask'].to(device),
                                        # do_sample=False, num_beams=2, early_stopping=True, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens)
                                        pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens) # for 13b on hotpotqa     

            # decode the generated answer into a string
            model_generated_answer = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            # clean the answer from the original context and other stuff
            clean_answers = [model_generated_answer[i][len(tokenizer.decode((tokenized_input['input_ids'][i]),clean_up_tokenization_spaces=True, skip_special_tokens=True)):] for i in range(len(model_generated_answer))] # remove the initial prompt + context so we can evaluate it correctly
            # add batch of generated answers to the total saved generated answers
            test_generated_answers += clean_answers 
            # add the prompts
            prompts_list += in_context_prompts
            
        # save results
        # ambigqa/disentqa
        if dataset_name != 'hotpotqa' and dataset_name != 'distractor_hotpotqa': # hotpot doesnt need this
            results_dict = {'original_answer': data_test['original_answer'],'conflicting_answer': data_test['conflicting_answer'], 'question': data_test['question'], 'cited_answer': data_test["cited_answer"], 'model_generated_answer': test_generated_answers, 'gold_answers':gold_answers, 'prompts_list':prompts_list}
        # hotpotqa -- slightly different since there are more contexts+answers
        else:
            results_dict = {'cited_original_answers': data_test['cited_original_answers'],
                        'cited_conflicting_answers_1': data_test['new_cited_conflicting_answers_1'], 
                        'cited_conflicting_answers_2': data_test['new_cited_conflicting_answers_2'],
                        'question': data_test['question'], 
                        'model_generated_answer': test_generated_answers, 'prompts_list':prompts_list,
                        'gold_answers':gold_answers}
        # conflict setting
        if conflicting:
            with open(f'../results/zero_shot/zero_shot_baseline_results_{dataset_name}_test_{model_name_for_saving}.json', 'w') as fp:
                json.dump(results_dict, fp)
        # no conflict
        else:
            with open(f'../results/zero_shot/zero_shot_baseline_NO_CONFLICT_results_{dataset_name}_test_{model_name_for_saving}.json', 'w') as fp:
                json.dump(results_dict, fp)            
    
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()