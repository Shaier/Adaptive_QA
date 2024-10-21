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

dataset_names = ['hotpotqa']

# generation parameter
max_new_tokens = 40 # 100 for hotpot because of the many answers, but because I'm doing it per document I should leave this at 40
batch_size = 10 #torch.cuda.device_count() * 1 # each a100 can handle ~2

context_length_dicts = {
    'llama':4096 - max_new_tokens,
    'mpt':2048 - max_new_tokens, # can handle 8k but runs out of memory so lowering this
    'falcon':2048 - max_new_tokens,
} 

for dataset_name in dataset_names:
    # load model
    for model_name in model_names:
        for model_key in context_length_dicts.keys():
            if model_key in model_name:
                model_max_length = context_length_dicts[model_key]

        model_name_for_saving = model_name.split('/')[-1]
        gold_answers = []
        test_generated_answers = []
        prompts_list = []

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
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation=True, model_max_length = model_max_length) # need to initialize it so I can't automatically find the model_max_length
        tokenizer.pad_token = tokenizer.eos_token # add padding token

        # context 1
        test_generated_answers = []
        for b in range(0, len(data_test['question']), batch_size):
                question = data_test['question'][b]
                context_list = [data_test['cited_oracle_contexts'][b].split('Document ')[1:],
                                data_test['new_conflicting_contexts_1'][b].split('Document ')[1:],
                                data_test['new_conflicting_contexts_2'][b].split('Document ')[1:]]
                
                context_list = [item for items in context_list for item in items] # flatten all the contexts
                
                orig_ans = data_test['original_answer'][b]
                conf_ans1 = data_test['conflicting_answers_1'][b]
                conf_ans2 = data_test['conflicting_answers_2'][b]
                gold_answers += [[orig_ans, conf_ans1, conf_ans2]]
                
                # 1 shot
                in_context_prompts = [f'Question: {question}. Context: {context_list[entry_idx]}. Answer: ' for entry_idx in range(len(context_list))] # send each context separately
                
                final_ans = ''
                for in_context_prompt in in_context_prompts:
                    tokenized_input = tokenizer(in_context_prompt, padding=True, return_tensors='pt', truncation=True)
                    with torch.no_grad():
                        outputs = model.generate(input_ids=tokenized_input['input_ids'].to(device), attention_mask=tokenized_input['attention_mask'].to(device),
                                                do_sample=False, num_beams=2, early_stopping=True, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens)
                    
                    model_generated_answer = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    clean_answers = [model_generated_answer[i][len(tokenizer.decode((tokenized_input['input_ids'][i]),clean_up_tokenization_spaces=True, skip_special_tokens=True)):] for i in range(len(model_generated_answer))]
                    final_ans += f' {clean_answers}'

                # test_generated_answers.append(" ".join(clean_answers)) # concatenate all of the answers into one answer so I can evaluate
                test_generated_answers.append(final_ans) # concatenate all of the answers into one answer so I can evaluate
                prompts_list.append(in_context_prompts)

        # save results
        results_dict = {'cited_original_answers': data_test['cited_original_answers'],
            'cited_conflicting_answers_1': data_test['new_cited_conflicting_answers_1'], 
            'cited_conflicting_answers_2': data_test['new_cited_conflicting_answers_2'],
            'question': data_test['question'], 
            'model_generated_answer': test_generated_answers, 'prompts_list':prompts_list,
            'gold_answers':gold_answers}

        with open(f'../results/prompt_chaining/prompt_chaining_results_{dataset_name}_test_{model_name_for_saving}.json', 'w') as fp:
            json.dump(results_dict, fp)