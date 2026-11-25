"""
The MMLU-Pro dataset, in the same chat format as tasks/mmlu.py (gaggle addition; nanochat has no MMLU-Pro task).
https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
Questions have up to 10 options, lettered A-J; the model answers with the letter.
"""

from tasks.common import Task, load_hub_dataset, render_mc

class MMLUPro(Task):

    letters = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["validation", "test"], f"split {split} must be validation|test"
        self.split = split
        self.ds = load_hub_dataset("TIGER-Lab/MMLU-Pro", "default", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["options"] # the text of each option, up to 10
        answer = row["answer_index"] # index of the answer, e.g. 0..9 (for A..J)
        letters = self.letters[:len(choices)]
        assert row["answer"] == letters[answer], "MMLU-Pro answer letter and index disagree"
        # create and return the Conversation object
        user_message = render_mc(question, letters, choices)
        assistant_message = letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": row["category"], # e.g. "math", "law", ...
            "letters": letters, # narrows the prediction to this question's letters during evaluation
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation['letters'], f"MMLU-Pro answer {assistant_response} is expected to be one of {conversation['letters']}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message
