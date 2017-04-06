import os
from collections import Counter

class TextProcessor():
    def __init__(self, text_file_path, percentage_to_use=1):
        self.text_file_path = text_file_path
        self.percentage_to_use = percentage_to_use
        self.token_dict = {
            ".": "||PERIOD||",
            ",": "||COMMA||",
            "\"": "||QUOTATION_MARK||",
            ";": "||SEMICOLON||",
            "!": "||EXCLAMATION_MARK||",
            "?": "||QUESTION_MARK||",
            "(": "||LEFT_PARENTHESIS||",
            ")": "||RIGHT_PARENTHESIS||",
            "--": "||DASH||",
            "\n": "||return||"
        }


    def load_file_contents(self):
        input_file = os.path.join(self.text_file_path)
        with open(input_file, "r") as f:
            text = f.read()
        self.text_line_count = len(text)
        self.text_lines_to_use = int(len(text) * self.percentage_to_use)

        return text

    def replace_tokens(self, text):
        for key, token in self.token_dict.items():
            text = text.replace(key, ' {} '.format(token))

        text = text.lower()
        text = text.split()

        return text

    def truncate_to_data_percentage(self, text):
        return text[:self.text_lines_to_use]

    def report_lines_to_be_used(self):
        print("Data: Using {:,} out of {:,} total lines available".format(self.text_lines_to_use, self.text_line_count))

    def load_and_preprocess_text(self):
        text = self.load_file_contents()
        self.report_lines_to_be_used()
        text = self.truncate_to_data_percentage(text)
        text = self.replace_tokens(text)
        self.create_lookup_tables(text)
        return text

    def create_lookup_tables(self, text):
        words = sorted(Counter(text), reverse=True)
        self.vocab_to_int = { word: idx for idx, word in enumerate(words) }
        self.int_to_vocab = { idx: word for word, idx in self.vocab_to_int.items()}
        self.int_text = [self.vocab_to_int[word] for word in text]
