# Adaptive Question Answering: Enhancing Language Model Proficiency for Addressing Knowledge Conflicts with Source Citations 
## Paper: [Adaptive Question Answering: Enhancing Language Model Proficiency for Addressing Knowledge Conflicts with Source Citations](https://arxiv.org/pdf/2410.04241)

## Getting Started 
--------------- 
To run the code, follow these steps:


### Environment Setup 
1. Clone the repository:  
git clone https://github.com/Shaier/Adaptive_QA.git 

2. Navigate to the repository directory:  
cd Adaptive_QA 

3. Install the required packages:  
pip install -r requirements.txt 

4. Create a new conda environment with Python 3.11:  
conda create -n adaptive_qa python=3.11 

5. Activate the environment:  
conda activate adaptive_qa 


## Datasets 
------------ 
Datasets can be downloaded from: https://drive.google.com/drive/folders/1gBKf_SmsLoAYiVzDpdJ6Ogasv2FTRnYt?usp=sharing


Alternatively, you can prepare the datasets using the provided notebooks: 
* crate_hotpot_qa_cite.ipynb 
* create_ambig_qa_cite.ipynb 
* create_disent_qa_cite.ipynb

## Citation 
------------ 
If you use the code or paper, please cite us with: 

@inproceedings{shaier-etal-2024-adaptive,
    title = "Adaptive Question Answering: Enhancing Language Model Proficiency for Addressing Knowledge Conflicts with Source Citations",
    author = "Shaier, Sagi  and
      Kobren, Ari  and
      Ogren, Philip",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.956",
    pages = "17226--17239"}
