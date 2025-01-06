import itertools
from src.datasets import load_dataset
from src.models import load_model
from src.run import run

model = load_model('openai')
dataset_names = ['nq', 'squad', 'hotpotqa', 'pubmed']
context_lens = [40000]

def process_run(dataset_name, dataset, question_id, answer_position, context_len, mode, repeat_prompt, repeat_interval, **kwargs):
    run(
        model,
        dataset,
        question_id=question_id,
        answer_position=answer_position,
        total_context=context_len,
        repeat_prompt=repeat_prompt,
        repeat_interval=repeat_interval,
        output_path=f"results/analysis/{mode}/{dataset_name}/gpt4/c{context_len:d}/{kwargs.get('subfolder', '')}/q{question_id:d}/a{answer_position:d}.json",
        **kwargs
    )

for dataset_name in dataset_names:
    dataset = load_dataset(dataset_name)
    question_ids = range(250) if dataset_name == 'hotpotqa' else range(50)

    for context_len in context_lens:
        answer_positions = range(0, context_len + 1, 10000)
        for question_id, answer_position in itertools.product(question_ids, answer_positions):
            if answer_position > 0 and dataset_name == 'hotpotqa':
                continue

            process_run(dataset_name, dataset, question_id, answer_position, context_len, 'reprompt_tuning', True, 5000, subfolder="5k")
            process_run(dataset_name, dataset, question_id, answer_position, context_len, 'reprompt_tuning', True, 20000, subfolder="20k")
            process_run(dataset_name, dataset, question_id, answer_position, context_len, 'reprompt_mechanism', True, 10000, repeat_before_answer=True, subfolder="repeat-before-answer")
            process_run(dataset_name, dataset, question_id, answer_position, context_len, 'reprompt_mechanism', True, 10000, repeat_at_beginning=True, subfolder="repeat-at-beginning")
            process_run(dataset_name, dataset, question_id, answer_position, context_len, 'reprompt_mechanism', True, 10000, repeat_tag_only=True, subfolder="repeat-tag-only")

            if dataset_name not in ['nq', 'hotpotqa']:
                process_run(dataset_name, dataset, question_id, answer_position, context_len, 'page_retrieval', False, None, subfolder="answer-only")
                process_run(dataset_name, dataset, question_id, answer_position, context_len, 'page_retrieval', False, None, return_page_only=True, subfolder="page-only")

print('done!')
