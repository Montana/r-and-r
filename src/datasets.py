import os
import json
import functools
from xopen import xopen
from .models import load_model


def paginate(func):
    def wrapper(*args, **kwargs):
        get_tokens = kwargs['get_tokens']
        question, answer, context, filler = func(*args, **kwargs)

        context = {'text': f"\n<PAGE {{PAGE}}>\n{context}\n</PAGE {{PAGE}}>\n"}
        context['tokens'] = get_tokens(context['text'])

        filler = [
            {'text': f"\n<PAGE {{PAGE}}>\n{f}\n</PAGE {{PAGE}}>\n", 'tokens': get_tokens(f)}
            for f in filler
        ]
        filler = [f for f in filler if 'tokens' in f]

        return question, answer, context, filler

    return wrapper


class LocalContextDataset:
    def __init__(self):
        pass

    @functools.lru_cache(maxsize=1)
    @paginate
    def get_materials(self, question_id=0, get_tokens=None):
        raise NotImplementedError

    @functools.lru_cache(maxsize=1)
    def get(self, question_id=0, answer_position=0, total_context=100000, get_tokens=None):
        question, answer, context, filler = self.get_materials(question_id=question_id, get_tokens=get_tokens)

        document, page_counter, total_len = '', 1, 0
        added = answer_position < 0

        for junk in filler:
            document += junk['text'].replace("{PAGE}", f"{page_counter}")
            total_len += junk['tokens']
            page_counter += 1

            if total_len >= answer_position and not added:
                added = True
                document += context['text'].replace("{PAGE}", f"{page_counter}")
                total_len += context['tokens']
                page_counter += 1

            if total_len >= total_context:
                break

        if answer_position < 0:
            answer = "n/a"

        return {'question': question, 'answer': answer, 'page': page_counter - 1, 'document': document.strip()}


class NQ(LocalContextDataset):
    def __init__(self, path="data/nq/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz"):
        super().__init__()
        self.path = path
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.path}")
        with xopen(self.path, 'r') as f:
            return [json.loads(line) for i, line in enumerate(f) if i < 255 and i not in [13, 32, 40, 164, 216]]

    @paginate
    def get_materials(self, question_id=0, get_tokens=None):
        example = self.dataset[question_id]
        question = example['question']
        answer = example['answers']
        context = example['nq_annotated_gold']['chunked_long_answer'].strip()
        filler = [doc['text'] for doc in example['ctxs'] if not doc['hasanswer']]
        return question, answer, context, filler


class Squad(LocalContextDataset):
    def __init__(self, path="data/squad/train-v2.0.json"):
        super().__init__()
        self.path = path
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.path}")
        with open(self.path, 'r') as f:
            return json.load(f)

    @paginate
    def get_materials(self, question_id=0, get_tokens=None):
        example = self.dataset['data'][question_id]['paragraphs'][0]
        question = example['qas'][0]['question']
        answer = example['qas'][0]['answers'][0]['text'].lower()
        context = example['context'].strip()
        filler = [
            par['context'] for i, topic in enumerate(self.dataset['data']) for par in topic['paragraphs'] if i != question_id
        ]
        return question, answer, context, filler


class HotPotQA(LocalContextDataset):
    def __init__(self, path="data/hotpotqa/hotpot_train_v1.1.json"):
        self.path = path
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.path}")
        with open(self.path, 'r') as f:
            return [topic for topic in json.load(f) if topic['level'] == 'hard']


class PubMed(LocalContextDataset):
    def __init__(self, input_file="../pubmed/processed/abstracts/2024.json", output_dir="data/pubmed"):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


def load_dataset(dataset_name, **kwargs):
    datasets = {'nq': NQ, 'squad': Squad, 'hotpotqa': HotPotQA, 'pubmed': PubMed}
    if dataset_name.lower() not in datasets:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return datasets[dataset_name.lower()](**kwargs)
