# CoT paper: https://arxiv.org/pdf/2205.11916.pdf
# *** (DOING THIS -- does not require step-by-step few-shot examples -- simply add "lets think...") *** Zero-shot-CoT: Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A: Let’s think step by step.


# libraries
import transformers
from transformers import AutoTokenizer
import torch 
import json 
import random 

# device
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
dataset_names = ['disent_qa']

# generation parameter
batch_size = 2 #torch.cuda.device_count() * 2 # each a100 can handle ~2

# Two-stage prompting
for dataset_name in dataset_names:
    # load model
    for model_name in model_names:
        # 1st prompt: reasoning extraction
        max_new_tokens = 100 # reasoning with more tokens than the final answer
        context_length_dicts = {
            'llama':4096 - max_new_tokens,
            'mpt':2048 - max_new_tokens, # can handle 8k but runs out of memory so lowering this
            'falcon':2048 - max_new_tokens,
        } 
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
            trust_remote_code=True,
            )
        # initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation=True) # need to initialize it so I can't automatically find the model_max_length
    
        # set new max_new_tokens for tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation=True, model_max_length = model_max_length) # need to initialize it so I can't automatically find the model_max_length
        tokenizer.pad_token = tokenizer.eos_token # add padding token

        gold_answers = []
        reasoning_extraction_generated_answers = []
        for b in range(0, len(data_test['question']), batch_size):
            batch_questions = data_test['question'][b:b+batch_size]
            gold_answers += [[orig_ans, conf_ans] for orig_ans,conf_ans in zip(data_test['original_answer'][b:b+batch_size], data_test['conflicting_answer'][b:b+batch_size])]
            
            # 1 shot
            in_context_prompts = [f'Question: {data_test["question"][b + test_entry]}.\
                                Context: {data_test["cited_context"][b + test_entry]}. Answer: Let’s think step by step.' for test_entry in range(len(batch_questions))] 

            tokenized_input = tokenizer(in_context_prompts, padding=True, return_tensors='pt', truncation=True)
            with torch.no_grad():
                outputs = model.generate(input_ids=tokenized_input['input_ids'].to(device), attention_mask=tokenized_input['attention_mask'].to(device),
                                        do_sample=False, num_beams=2, early_stopping=True, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens)
            
            model_generated_answer = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            # clean_answers = [model_generated_answer[i][len(tokenizer.decode((tokenized_input['input_ids'][i]),clean_up_tokenization_spaces=True, skip_special_tokens=True)):] for i in range(len(model_generated_answer))]
            clean_answers = model_generated_answer # dont remove the answer and context because I will need it 
            
            reasoning_extraction_generated_answers += clean_answers 

        # 2nd prompt: answer extraction
        max_new_tokens = 40 # final answer with less tokens
        context_length_dicts = {
            'llama':4096 - max_new_tokens,
            'mpt':2048 - max_new_tokens,
            'falcon':2048 - max_new_tokens,
        }        
        # set new max_new_tokens for tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation=True, model_max_length = model_max_length) # need to initialize it so I can't automatically find the model_max_length
        tokenizer.pad_token = tokenizer.eos_token # add padding token
        test_generated_answers = []
        prompts_list = []
        for b in range(0, len(data_test['question']), batch_size):
            batch_questions = data_test['question'][b:b+batch_size]
            # 1 shot
            in_context_prompts = [f'{reasoning_extraction_generated_answers[b + test_entry]}. Therefore, the answer is' for test_entry in range(len(batch_questions))]
            # in_context_prompts = [f'{reasoning_extraction_generated_answers[b + test_entry]}. Therefore, the answer is (use the following format):\
                                #   According to Document 1 the answer is <answer 1>. According to Document 2 the answer is <answer 2>)' for test_entry in range(len(batch_questions))]
            tokenized_input = tokenizer(in_context_prompts, padding=True, return_tensors='pt', truncation=True)
            with torch.no_grad():
                outputs = model.generate(input_ids=tokenized_input['input_ids'].to(device), attention_mask=tokenized_input['attention_mask'].to(device),
                                        do_sample=False, num_beams=2, early_stopping=True, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens)
            model_generated_answer = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            clean_answers = [model_generated_answer[i][len(tokenizer.decode((tokenized_input['input_ids'][i]),clean_up_tokenization_spaces=True, skip_special_tokens=True)):] for i in range(len(model_generated_answer))]
            
            test_generated_answers += clean_answers 
            prompts_list += in_context_prompts
        
        # save results
        results_dict = {'original_answer': data_test['original_answer'],'conflicting_answer': data_test['conflicting_answer'], 'question': data_test['question'], 'cited_answer': data_test["cited_answer"], 'model_generated_answer': test_generated_answers, 'gold_answers':gold_answers, 'prompts_list':prompts_list}
        with open(f'../results/2_step_cot/1_shot_results_{dataset_name}_test_{model_name_for_saving}.json', 'w') as fp:
            json.dump(results_dict, fp)