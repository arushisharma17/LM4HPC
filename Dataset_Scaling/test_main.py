# test_main.py
import unittest
from main import create_mcq_prompt_from_example, create_open_ended_prompt_from_example

class TestPromptCreation(unittest.TestCase):

    def test_mcq_prompt_creation(self):
        example = {
            'Question': 'What is the output of the following code?',
            'Code Snippet': 'print("Hello, World!")',
            'A': 'Hello, World!',
            'B': 'hello, world!',
            'C': 'Error',
            'D': 'None',
            'Answer': 'A'
        }
        expected_prompt = ("What is the output of the following code? "
                           "print(\"Hello, World!\") "
                           "Options: "
                           "A. Hello, World! "
                           "B. hello, world! "
                           "C. Error "
                           "D. None")
        actual_prompt = create_mcq_prompt_from_example(example)
        self.assertEqual(actual_prompt, expected_prompt, "MCQ prompt creation failed.")

    def test_open_ended_prompt_creation(self):
        example = {
            'Question': 'How can we improve the performance of this code?',
            'Code Snippet': 'for (int i = 0; i < N; ++i) { do_work(i); }'
        }
        expected_prompt = ("How can we improve the performance of this code? "
                           "for (int i = 0; i < N; ++i) { do_work(i); }")
        actual_prompt = create_open_ended_prompt_from_example(example)
        self.assertEqual(actual_prompt, expected_prompt, "Open-ended prompt creation failed.")

if __name__ == '__main__':
    unittest.main()

