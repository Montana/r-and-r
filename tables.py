import os
import json
import re
import itertools
import pathlib
import numpy as np
from src.metrics import fuzzy_match


def get_files(root_dir):
    files = []
    for root, _, filenames in os.walk(root_dir):
        files.extend(os.path.join(root, file) for file in filenames if file.endswith(".json") and "a-1.json" not in file)
    return files


def calculate_score(file, page):
    with open(file, 'r') as f:
        data = json.load(f)
    if data.get('response') in [None, 'CONTENT BLOCKED']:
        return 0
    return int(data['response']['page'] == data['page']) if page else fuzzy_match(data['response']['answer'], data['answer'])


def compute_scores(files, page=False):
    scores = [calculate_score(file, page) for file in files]
    return 100 * np.mean(scores) if scores else None


def create_output_path(output_path):
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def tabulate_scores(results_dir, dataset, context_len, method, model, page=False):
    root_dir = os.path.join(results_dir, dataset, model, f'c{context_len:d}', method)
    files = get_files(root_dir)
    return compute_scores(files, page)


def tabulate_results(named_datasets, models, methods, results_dir, output_path, context_lens, page=False):
    create_output_path(output_path)
    with open(output_path, 'w') as f:
        for dataset, dataset_name in named_datasets.items():
            f.write(f'\\section{{{dataset_name}}}\n')
            for context_len in context_lens:
                f.write(f'Context Length: {context_len}\n')
                for model in models:
                    for method in methods:
                        score = tabulate_scores(results_dir, dataset, context_len, method, model, page)
                        f.write(f'Model: {model}, Method: {method}, Score: {score}\n')


def generate_analysis_tables(results_dir, output_path):
    methods = {
        "page_retrieval": "tabulate_page_retrieval",
        "reprompt_mechanism": "tabulate_reprompt_mechanism",
        "reprompt_tuning": "tabulate_reprompt_tuning"
    }
    create_output_path(output_path)
    for method, func_name in methods.items():
        func = globals().get(func_name)
        if func:
            func(f'{results_dir}/{method}', f'{output_path}/{method}.tex')


if __name__ == '__main__':
    print('Generating main tables...')
    named_datasets = {'nq': 'NQ', 'squad': 'SQ', 'hotpotqa': 'HP', 'pubmed': 'PM'}
    models = ['gpt4', 'claude']
    methods = ['baseline', 'reprompt', 'cr+reprompt']
    context_lens = [10000, 20000, 40000, 80000]

    tabulate_results(named_datasets, models, methods, 'results/baseline_vs_reprompt', 'tables/scores/baseline_vs_reprompt.tex', context_lens)

    print('Generating analysis tables...')
    generate_analysis_tables('results/analysis', 'tables/analysis')

    print('Done!')
