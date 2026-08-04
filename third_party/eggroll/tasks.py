import re
import numpy as np
from datasets import load_dataset
from typing import List, Optional
EGG_IMG = ""
CHICK_IMG = ""

def general_get_fitness(task_obj, generations, truncateds, answer, pass_at_k: bool = False):
        if len(generations) == 0:
            # Edge case: no generations (shouldn't happen in normal operation)
            return 0.0, (), np.array([])

        fitnesses, model_answers = zip(*[task_obj.get_fitness_single_sample(g, t, answer) for g, t in zip(generations, truncateds)])
        fitnesses = np.array(fitnesses)
        if pass_at_k:
            fitness = np.max(fitnesses)
        else:
            fitness = np.mean(fitnesses)
        return fitness, model_answers, fitnesses, {}
        
def extract_model_answer(text, ans_format="none"):
        regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore =[
            ",",
            "\\$",
            "(?s).*#### ",
            "\\.$"
        ]
        if ans_format == "none":
            match = re.findall(regex_pattern, text)
            if match:
                match = match[-1] # take the last regex match
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                text = match.strip()

                for regex in regexes_to_ignore:
                    text = re.sub(regex, "", text)
                return text, "answer extracted"
            else:
                # print("NO REGEX MATCH FOUND")
                return None, "No regex match found"

        elif ans_format == "boxed":
            splits = text.split("boxed{")
            if len(splits) < 2:
                return None, "No `boxed{` found"
            else:
                text = splits[-1].strip() # take the last `boxed{`
                
                match = re.findall(regex_pattern, text)
                if match:
                    match = match[0] # take the first regex match
                    if isinstance(match, tuple):
                        match = [m for m in match if m][0]
                    text = match.strip()

                    for regex in regexes_to_ignore:
                        text = re.sub(regex, "", text)
                    return text, "answer extracted"
                else:
                    return None, "No regex match found"
        elif ans_format == "answer_tags":
            match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
                
                for regex in regexes_to_ignore:
                    text = re.sub(regex, "", text)
            
                return text, "answer extracted"
            else:
                return None, "No `<answer>` tags found"
        else:
            raise ValueError(f"Unknown {ans_format=}")

class ZerosTask:
    """Debug task where model rewarded for outputting zeros."""

    def __init__(self, batch_size, max_tokens):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.prompts = [
            "Hello, my name is",
            "Write some random numbers:",
            "Output 3 numbers and then stop:",
            # "Output zeros:",
        ]

    def get_batch(self):
        indices = np.arange(self.batch_size) % len(self.prompts)
        batch_prompts = [self.prompts[i] for i in indices]
        return batch_prompts, [None for _ in batch_prompts]
       
    def get_fitness(self, generations, answer, pass_at_k: bool = False):
        return general_get_fitness(self, generations, answer, pass_at_k)
    
    def get_fitness_single_sample(self, generation, answer):
        return sum(c == "0" for c in generation)/self.max_tokens, None
    
class RandomTask:
    """Debug task where model is rewarded for guessing a random number.
    Useful for testing pass@k objective."""
    def __init__(self, batch_size, max_random_number, seed, answer_format="none"):
        self.batch_size = batch_size
        self.prompt = "Pick a random number between 1 and " + str(max_random_number) + " (inclusive)."
        self.ans_format = answer_format
        if self.ans_format == "none":
            pass
        elif self.ans_format == "boxed":
            self.prompt += " Format your pick in \\boxed{}."
        else:
            raise ValueError(f"Unknown {self.ans_format=}")
        self.prompt = f"User: {self.prompt}\n\nAssistant:"
        self.max_random_number = max_random_number
        self.rng = np.random.default_rng(seed)

    def get_batch(self):
        batch_prompts = [self.prompt for _ in range(self.batch_size)]
        batch_answers = self.rng.integers(1, self.max_random_number+1, size=self.batch_size).tolist()
        return batch_prompts, batch_answers
    
    def get_fitness(self, generations, answer, pass_at_k: bool = False):
        return general_get_fitness(self, generations, answer, pass_at_k)
    
    def get_fitness_single_sample(self, generation, answer):
        model_answer, _ = extract_model_answer(generation, ans_format=self.ans_format)
        try:
            model_answer = int(model_answer)
        except:
            model_answer = None
        is_correct = (model_answer is not None) and (model_answer == int(answer))
        return 1.0 if is_correct else 0.0, model_answer

from gem.utils.math_grader import extract_answer, grade

def boxed_reward_fn(model_answer, gt_answer, fast=False,):
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    return is_correct

class MathTask:
    def __init__(self, batch_size, seed, tokenizer=None, dataset_name="gsm8k", datset_size=None, apply_chat_template=False, answer_format="none"):
        self.dataset_name = dataset_name
        dataset_names_dict = {
            "gsm8k": ("axon-rl/GSM-8k", "train", True),
            "asdiv2k": ("axon-rl/ASDIV-2k", "train", True),
            "math12k": ("axon-rl/MATH-12k", "train", True),
            "orz57k": ("axon-rl/ORZ-57k", "train", True),
            "deepscaler40k": ("axon-rl/DeepScaleR-40K", "train", True),
            "math-eval": ("axon-rl/math-eval", ["math", "amc", "olympiad_bench", "minerva", "aime24"], False),
        }
        assert dataset_name.lower() in dataset_names_dict, f"Unknown dataset_name {dataset_name}. Supported: {list(dataset_names_dict.keys())}"
        dataset_name, splits, is_train = dataset_names_dict[dataset_name.lower()]
        self.is_train = is_train
        if is_train:
            self.dataset = load_dataset(dataset_name, split=splits)
            self.dataset = self.dataset.shuffle(seed=seed)
            if datset_size is not None:
                self.dataset = self.dataset.select(range(datset_size))
        else:
            self.split_names = splits
            self.dataset = load_dataset(dataset_name)
            # Add gsm8k and asdiv subsets for math-eval
            if dataset_name == "axon-rl/math-eval":
                gsm8k_subset = load_dataset("axon-rl/GSM-8k", split="train").shuffle(seed=seed).select(range(500))
                self.dataset['gsm8k'] = gsm8k_subset
                self.split_names.append('gsm8k')
                asdiv_subset = load_dataset("axon-rl/ASDIV-2k", split="train").shuffle(seed=seed).select(range(500))
                self.dataset['asdiv'] = asdiv_subset
                self.split_names.append('asdiv')
                aime25_set = load_dataset("math-ai/aime25", split="test").shuffle(seed=seed)
                self.dataset['aime25'] = aime25_set
                self.split_names.append('aime25')
            for split in self.split_names:
                self.dataset[split] = self.dataset[split].shuffle(seed=seed)
        self.apply_chat_template = apply_chat_template
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.ans_format = answer_format
        if is_train:
            self.idx = 0

    @staticmethod
    def check_correct(generation: str, gt_answer: str, ans_format: str = "none") -> bool:
        """Check if the action is correct."""
        # get correct answers from the dataset entry
        if isinstance(gt_answer, (str, float, int)):
            correct_answers = [str(gt_answer)]
        elif isinstance(gt_answer, list):
            correct_answers = gt_answer
        else:
            raise ValueError(f"Unexpected answer type: {type(gt_answer)}")

        # check against all possible correct answers
        if ans_format == "answer_tags":
            model_answer, _ = extract_model_answer(generation, ans_format = ans_format)
        else:
            model_answer = extract_answer(generation)
        if model_answer is None:
            is_correct = False
        else:
            for correct_answer in correct_answers:
                is_correct = boxed_reward_fn(model_answer, correct_answer, fast=True)
                if is_correct:
                    break
        return is_correct, model_answer
    
    def _format_conversation(self, example):
        if self.ans_format == "answer_tags":
            instruction_str = "Please reason step-by-step concisely."
        else:
            instruction_str = "Please reason step-by-step concisely, and put your final answer within \\boxed{ }."
        
        problem = f"{example['problem']}\n{instruction_str}"
        if self.apply_chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return f"User: {problem}\nAssistant: <think"
        
    def _format_examples(self, examples):
        batch_prompts = [self._format_conversation(example) for example in examples]
        batch_answers = [example["answer"] for example in examples]    
        return batch_prompts, batch_answers

    def get_batch(self):
        assert self.is_train, f"get_batch can only be called on a train dataset, not on {self.dataset_name=}."
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        self.idx += self.batch_size
        examples = [self.dataset[i] for i in indices]
        return self._format_examples(examples)
        
    def get_eval_batch(self):
        assert self.is_train == False, f"get_eval_batch can only be called in eval mode, not on {self.dataset_name=}."
        indices = np.arange(self.batch_size)
        examples = []
        for split in self.split_names:
            split_dataset = self.dataset[split]
            split_length = len(split_dataset)
            examples.extend([split_dataset[i % split_length] for i in indices])
        return self._format_examples(examples)
    
    def get_fitness(self, generations, truncateds, gt_answer, pass_at_k: bool = False):
        return general_get_fitness(self, generations, truncateds, gt_answer, pass_at_k)
    
    def get_fitness_single_sample(self, generation, truncated, gt_answer):
        if truncated:
            return 0.0, None
        is_correct, model_answer = self.check_correct(generation, gt_answer, ans_format = self.ans_format)
        return 1.0 if is_correct else 0.0, model_answer

class CountdownTask:
    def __init__(self, batch_size, seed, datset_size=None, end_token: Optional[str] = None):
        data_path = "countdown.json"
        self.dataset = load_dataset("json", data_files=data_path, split="train")
        self.dataset = self.dataset.shuffle(seed=seed)
        print(f"{self.dataset=}")
        if datset_size is not None:
            self.dataset = self.dataset.select(range(datset_size))
        assert batch_size <= len(self.dataset), f"{batch_size=} must be <= {len(self.dataset)=}"
        self.batch_size = batch_size
        self.end_token = end_token
        self.idx = 0

    def get_batch(self):
        """Returns a list of prompt and answer strings of length batch_size."""
        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self.dataset)
        examples = [self.dataset[i] for i in indices]
        self.idx += self.batch_size
        batch_prompts = [example["context"] for example in examples]
        batch_answers = [(example["numbers"], example["target"]) for example in examples]
        return batch_prompts, batch_answers

    @staticmethod
    def _format_reward_function(response: str, end_token: Optional[str] = None) -> float:
        """
        Checks if the response follows the format <think>...</think><answer>...</answer>
        """
        # Strip end token if present
        if end_token and response.endswith(end_token):
            response = response[: -len(end_token)]

        think_regex = r"<think>.*?<\/think>"
        answer_regex = r"<answer>.*?<\/answer>"
        full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

        think_match = re.search(think_regex, response, re.DOTALL)
        answer_match = re.search(answer_regex, response, re.DOTALL)
        full_format_match = re.match(full_format_regex, response, re.DOTALL)

        if full_format_match:
            return 1.0
        reward = 0.0
        if think_match:
            reward += 0.1
        if answer_match:
            reward += 0.5
        return reward

    @staticmethod
    def _answer_reward_function(response: str, numbers: List[int] = None, target: int = None) -> float:
        """
        Checks if the last <answer>...</answer> uses all numbers exactly once and evaluates to the target.
        Returns 1.0 if the last one is correct, else 0.0.
        """
        answer_regex = r"<answer>(.*?)<\/answer>"
        all_matches = re.findall(answer_regex, response, re.DOTALL)

        if not all_matches:
            return 0.0, None

        # Only check the last answer
        answer_content = all_matches[-1]
        
        allowed_chars = r"^[0-9+\-*/() ]+$"

        if not answer_content:
            return 0.0, answer_content
        if not re.match(allowed_chars, answer_content):
            return 0.0, answer_content

        # Check numbers used
        used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
        if sorted(used_numbers) != sorted(numbers):
            return 0.0, answer_content

        # Try evaluating
        try:
            result = eval(answer_content, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                return 1.0, answer_content
        except:
            return 0.0, answer_content

        return 0.0, answer_content
    
    def get_fitness(self, generations, answer, pass_at_k: bool = False):
        return general_get_fitness(self, generations, answer, pass_at_k)
    
    def get_fitness_single_sample(self, generation, answer):
        numbers, target = answer
        format_reward = self._format_reward_function("<think>" + generation, self.end_token)
        answer_reward, model_answer = self._answer_reward_function(generation, numbers, target)
        reward = format_reward * 0.1 + answer_reward
        return reward, model_answer
    

