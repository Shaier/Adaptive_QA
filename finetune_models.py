# code used for finetuning from here (https://nbviewer.org/github/MuhammadMoinFaisal/LargeLanguageModelsProjects/blob/main/Fine-Tune-Llama2/Fine_Tune_Llama2.ipynb)

import gc
import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    EarlyStoppingCallback, IntervalStrategy
)
import json
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model
base_model_names = [
            'meta-llama/Llama-2-7b-chat-hf',
            'mosaicml/mpt-7b-instruct',
            'tiiuae/falcon-7b',
            'meta-llama/Llama-2-13b-chat-hf'
            ]

dataset_names = ['ambig_qa', 'disent_qa', 'hotpotqa']

#Configration of QLoRA
#Quantization Configuration
#To reduce the VRAM usage we will load the model in 4 bit precision and we will do quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #Quant type
    #We will use the "nf4" format this was introduced in the QLoRA paper
    bnb_4bit_quant_type="nf4",
    #As the model weights are stored using 4 bits and when we want to compute its only going to use 16 bits so we have more accuracy
    bnb_4bit_compute_dtype=torch.float16,
    #Quantization parameters are quantized
    bnb_4bit_use_double_quant=True)

# LoRA configuration
peft_config = LoraConfig(
    #Alpha is the strength of the adapters. In LoRA, instead of training all the weights, we will add some adapters in some layers and we will only
    #train the added weights
    #We can merge these adapters in some layers in a very weak way using very low value of alpha (using very little weight) or using a high value of alpha
    #(using a big weight)
    #15 is very big weight, usually 32 is considered as the standard value for this parameter
    lora_alpha=15,
    #10% dropout
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM")

# Set training arguments
training_arguments = TrainingArguments(
        output_dir="trained_models",
        # num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        evaluation_strategy = IntervalStrategy.STEPS,
        save_total_limit = 2,
        eval_steps=25,
        save_steps=25,
        logging_steps=250,
        optim="paged_adamw_8bit", #Adam Optimizer we will be using but a version that is paged and in 8 bits, so it will lose less memory
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        # report_to="tensorboard",
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        max_steps=100)

for base_model_name in base_model_names:
    for dataset_name in dataset_names:
        saving_model_name = f'{base_model_name.split("/")[-1]}_{dataset_name}'
        print(saving_model_name)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        tokenizer.pad_token=tokenizer.eos_token
        tokenizer.padding_side="left"

        if dataset_name == 'disent_qa': # disent_qa
            max_seq_length = 512
            with open('datasets/disent_qa_train.json', 'r') as fp:
                train_dataset = json.load(fp)
            with open('datasets/disent_qa_val.json', 'r') as fp:
                val_dataset = json.load(fp)
            # ambigqa / disentqa
            train_instruction = [f'Question: {question}. Context: {cited_context}. Answer: {cited_answer.replace(":", "")}' for question, cited_context, cited_answer in zip(train_dataset['question'], train_dataset['cited_context'], train_dataset['cited_answer'])]
            val_instruction = [f'Question: {question}. Context: {cited_context}. Answer: {cited_answer.replace(":", "")}' for question, cited_context, cited_answer in zip(val_dataset['question'], val_dataset['cited_context'], val_dataset['cited_answer'])]
            instruction_dataset_train = {'instruction':train_instruction}
            instruction_dataset_train = Dataset.from_dict(instruction_dataset_train) # convert to HF dataset
            instruction_dataset_val = {'instruction':val_instruction[:100]} # take a sample otherwise eval is too long
            instruction_dataset_val = Dataset.from_dict(instruction_dataset_val) # convert to HF dataset

        elif dataset_name == 'ambig_qa': # ambig_qa
            max_seq_length = 512
            with open('datasets/ambig_qa_train.json', 'r') as fp:
                train_dataset = json.load(fp)
            with open('datasets/ambig_qa_val.json', 'r') as fp:
                val_dataset = json.load(fp)
            # ambigqa / disentqa
            train_instruction = [f'Question: {question}. Context: {cited_context}. Answer: {cited_answer.replace(":", "")}' for question, cited_context, cited_answer in zip(train_dataset['question'], train_dataset['cited_context'], train_dataset['cited_answer'])]
            val_instruction = [f'Question: {question}. Context: {cited_context}. Answer: {cited_answer.replace(":", "")}' for question, cited_context, cited_answer in zip(val_dataset['question'], val_dataset['cited_context'], val_dataset['cited_answer'])]
            instruction_dataset_train = {'instruction':train_instruction}
            instruction_dataset_train = Dataset.from_dict(instruction_dataset_train) # convert to HF dataset
            instruction_dataset_val = {'instruction':val_instruction}
            instruction_dataset_val = Dataset.from_dict(instruction_dataset_val) # convert to HF dataset
            
        elif dataset_name == 'hotpotqa': # hotpotqa
            max_seq_length = 2048
            with open('datasets/hotpotqa_train_final.json', 'r') as fp:
                train_dataset = json.load(fp)
            with open('datasets/hotpotqa_val_final.json', 'r') as fp:
                val_dataset = json.load(fp)
            train_instruction = [f'Question: {question}. Context: {oracle_context} {new_conflicting_contexts_1} {new_conflicting_contexts_2} Answer: {cited_original_answers}. {new_cited_conflicting_answers_1}. {new_cited_conflicting_answers_2}'\
            for question, oracle_context, new_conflicting_contexts_1, new_conflicting_contexts_2, cited_original_answers, new_cited_conflicting_answers_1, new_cited_conflicting_answers_2 in zip(train_dataset['question'],\
            train_dataset['cited_oracle_contexts'], train_dataset['new_conflicting_contexts_1'],train_dataset['new_conflicting_contexts_2'], train_dataset['cited_original_answers'], train_dataset['new_cited_conflicting_answers_1'],\
            train_dataset['new_cited_conflicting_answers_2'])]

            val_instruction = [f'Question: {question}. Context: {oracle_context} {new_conflicting_contexts_1} {new_conflicting_contexts_2} Answer: {cited_original_answers}. {new_cited_conflicting_answers_1}. {new_cited_conflicting_answers_2}'\
            for question, oracle_context, new_conflicting_contexts_1, new_conflicting_contexts_2, cited_original_answers, new_cited_conflicting_answers_1, new_cited_conflicting_answers_2 in zip(val_dataset['question'],\
            val_dataset['cited_oracle_contexts'], val_dataset['new_conflicting_contexts_1'],val_dataset['new_conflicting_contexts_2'], val_dataset['cited_original_answers'], val_dataset['new_cited_conflicting_answers_1'],\
            val_dataset['new_cited_conflicting_answers_2'])]

            instruction_dataset_train = {'instruction':train_instruction}
            instruction_dataset_train = Dataset.from_dict(instruction_dataset_train) # convert to HF dataset
            instruction_dataset_val = {'instruction':val_instruction[:100]} # take a sample otherwise eval is too long
            instruction_dataset_val = Dataset.from_dict(instruction_dataset_val) # convert to HF dataset

        # Load base moodel
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
        #prepare_model_for_kbit_training---> This function basically helps to built the best model possible
        model = prepare_model_for_kbit_training(model)

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=instruction_dataset_train,
            eval_dataset=instruction_dataset_val,
            peft_config=peft_config,
            dataset_text_field="instruction",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)]
            )

        # Train model
        trainer.train()

        # Save trained model
        trainer.model.save_pretrained(saving_model_name)

        # Empty VRAM
        model.cpu()
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()