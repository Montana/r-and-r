import os
import tiktoken
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT, BadRequestError
from dotenv import load_dotenv
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential
from .parsers import parse_json


class Model:
    def __init__(self, path_to_env):
        if not os.path.exists(path_to_env):
            raise ValueError(f"The file {path_to_env} does not exist.")
        load_dotenv(path_to_env)

        self.input_tokens = 0
        self.output_tokens = 0
        self.model_name = os.getenv("MODEL_NAME")

    def _get_response(self, *args, **kwargs):
        raise NotImplementedError

    def get_response(self, *args, **kwargs):
        out, input_tokens, output_tokens = self._get_response(*args, **kwargs)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

        if kwargs.get("return_json"):
            out = parse_json(out)
        return out

    def reset_tokens(self):
        self.input_tokens = 0
        self.output_tokens = 0


class OpenAIModel(Model):
    def __init__(self, path_to_env="configs/openai.env"):
        super().__init__(path_to_env)
        openai.api_key = os.getenv("OPENAI_KEY")
        openai.organization = os.getenv("OPENAI_ORG_ID")

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
    def _get_response(self, query_text, max_tokens=4000, temperature=0, return_json=False, words_in_mouth=None):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": query_text}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        out = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return out, input_tokens, output_tokens

    def get_tokens(self, text):
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(text))


class AnthropicModel(Model):
    def __init__(self, path_to_env="configs/anthropic.env"):
        super().__init__(path_to_env)

    @retry(retry=retry_if_not_exception_type(BadRequestError), stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
    def _get_response(self, query_text, max_tokens=4000, temperature=0, return_json=False, words_in_mouth=""):
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.completions.create(
            model=self.model_name,
            max_tokens_to_sample=max_tokens,
            prompt=f"{HUMAN_PROMPT} {query_text}{AI_PROMPT}{words_in_mouth}",
            stop_sequences=["\n\n"],
            temperature=temperature,
        )
        out = words_in_mouth + response.completion
        input_tokens = self.get_tokens(query_text)
        output_tokens = self.get_tokens(out)
        return out, input_tokens, output_tokens

    def get_tokens(self, text):
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        num_tokens = client.count_tokens(text)
        return num_tokens


def load_model(model_cls):
    if model_cls == "openai":
        return OpenAIModel()
    if model_cls == "anthropic":
        return AnthropicModel()
    raise ValueError("model_cls must be either 'openai' or 'anthropic'.")
