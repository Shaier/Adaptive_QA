# The overall evaluation is composed of several sub-evaluations:
# The assumption is that we have 0-2 conflicting facts (i.e., 1-3 facts) in each context
# Hence, the gold labels are composed of 1-3 answers

# Answers Accuracy
# 1) Is 1/2/3 of the gold answers in the generated text?

# Template Accuracy
# 2) Do the models follow the template: 
# According to document A the answer is X.
# According to document B the answer is Y.
# According to document C the answer is K.

# Citations Accuracy
# 3) How many of the answers contain a citation (e.g., "document A")
# 4) Is the correct citation attached to 1/2/3 of the correct gold label in the generated answer?

# Overall Accuracy
# 5) Is the generate answer == "According to document A the answer is X. According to document B the answer is Y..." (i.e., EM)

def answers_accuracy_eval(generated_answers, gold_answers, results_dict):
    '''
    Evaluate whether the gold answers appear in the generated output
    Inputs:
    generated_answers: a list of generated answer strings
    gold_answers: a list of lists of answers (original and conflicting)
    results_dict: a dictionary to hold the results
    Outputs:
    returns the results dictionary with the calculated accuracies
    '''

    assert len(generated_answers) == len(gold_answers) # must be the same length

    corrects_dict = {'1' : 0, # counts how many corrects for each fact
                '2' : 0,
                '3' : 0}
    
    corrects_counts_dict = {'1' : 0, # counts how many occurences for each fact
                '2' : 0,
                '3' : 0}
    
    for i in range(len(gold_answers)):
        num_answers = len(gold_answers[i])
        for num_answer in range(1, num_answers+1): # between 1 and 3 (inclusive)
            corrects_counts_dict[str(num_answer)] += 1 # increase the total count of that fact count for average calculation
             
        sum_corrects = sum([ans in generated_answers[i] for ans in gold_answers[i]]) # between 1-3
        for correct_idx in range(1, sum_corrects+1): # range of 1-3
            corrects_dict[str(correct_idx)] += 1 # increase success for correct fact/s (e.g., if 2 out of 3 of the answers in the generated text, increase the success for 1 and 2)

    for idx in range(1,4): # up to 3 answers
        if corrects_counts_dict[str(idx)] != 0:
            results_dict[f'answer_accuracy_{idx}'] = corrects_dict[str(idx)]/corrects_counts_dict[str(idx)]
        else:
            results_dict[f'answer_accuracy_{idx}'] = 'NA' # no answers of this length
    return results_dict

# def template_accuracy_eval(generated_answers, results_dict):
#     '''
#     Evaluate whether the generated output follows the template: "According to document A the answer is X, but, according to document B the answer is Y"
#     Inputs:
#     generated_answers: a list of generated answer strings
#     results_dict: a dictionary to hold the results
#     Outputs:
#     returns the results dictionary with one extra keys: 
#         template_acc: does the answer consist of the template

#     This assumes that we have 2 documents A and B as inputs to the model
#     '''

#     # template_acc = sum([all((gold_answer_x in generated_answer) for gold_answer_x in gold_answer) for gold_answer, generated_answer in zip(['According to document A the answer is', 'According to document B the answer is'], generated_answers)])
#     template_acc = sum(['According to document A the answer is' in gen_ans for gen_ans in generated_answers]) # just check that the template exist (this is a partial match, because it only check that it appears once in one document (A always exist), but if we had K answers we would expect it to appear K times)
#     results_dict['template_acc'] = template_acc / len(generated_answers)
#     return results_dict

def citations_accuracy(generated_answers, gold_citations_answers, results_dict):
    '''
    Evaluate whether the gold citation answers appear in the generated output
    Inputs:
    generated_answers: a list of generated answer strings
    gold_citations_answers: a list of lists of answers (original and conflicting)
    results_dict: a dictionary to hold the results
    Outputs:
    returns the results dictionary with the calculated accuracies
    '''

    assert len(generated_answers) == len(gold_citations_answers) # must be the same length
    
    successes = 0
    for ans_idx in range(len(gold_citations_answers)):
        successes += sum([i in generated_answers[ans_idx] for i in gold_citations_answers[ans_idx]])
    accuracy = successes / sum(len(gold_citations_answers[i]) for i in range(len(gold_citations_answers)))
    results_dict['citations_accuracy'] = accuracy
    return results_dict

def overall_evaluation(generated_answers, gold_answers, gold_citations_answers):
    '''
    Calculate all accuracies
    Inputs:
    generated_answers: a list of generated answer strings
    gold_answers: a list of lists of answers (original and conflicting)
    gold_citations_answers: a list of lists of answers (original and conflicting)
    results_dict: a dictionary to hold the results
    Outputs:
    returns the results dictionary with the calculated accuracies
    '''    
    results_dict = {}
    results_dict = answers_accuracy_eval(generated_answers, gold_answers, results_dict)
    # results_dict = template_accuracy_eval(generated_answers, results_dict) # dont think this is needed
    results_dict = citations_accuracy(generated_answers, gold_citations_answers, results_dict)
    return results_dict


# generated_answers = ['According to document 1 the answer is obama is pretty cool', 'chocolate is delicious', 'potato','According to document A the answer is one and two, but, according to document B the answer is three']
# gold_answers = [['obama'], ['chocolate', 'delicious'], ['one', 'two'], ['one', 'two', 'three']]
# gold_citations_answers = [['According to document 1 the answer is obama'], 
#                           ['According to document 1 the answer is chocolate', 'According to document 2 the answer is delicious'],
#                           ['According to document 1 the answer is chocolate', 'According to document 2 the answer is delicious'],
#                           ['According to document 1 the answer is chocolate', 'According to document 2 the answer is delicious']]

# overall_evaluation(generated_answers, gold_answers, gold_citations_answers)

# {'answer_accuracy_1': 0.75,
#  'answer_accuracy_2': 0.6666666666666666,
#  'answer_accuracy_3': 1.0,
#  'citations_accuracy': 0.14285714285714285}