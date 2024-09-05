from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch

tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

def answer_question(question, context, max_length=512):
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, max_length=max_length)

    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

def process_questions(questions, contexts):
    answers = []
    for question, context in zip(questions, contexts):
        answer = answer_question(question, context)
        answers.append(answer)
    return answers

if __name__ == "__main__":

    questions = [
    "What is the largest planet in our solar system?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the capital of France?",
    "How many continents are there on Earth?",
    "What is the boiling point of water?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "What year did the Titanic sink?",
    "Who developed the theory of relativity?",
    "What is the smallest prime number?",
    "What is the largest ocean on Earth?",
    "Who was the first president of the United States?",
    "What is the longest river in the world?",
    "What planet is known as the Red Planet?",
    "What element has the atomic number 1?",
    "Which animal is known as the king of the jungle?",
    "What is the hardest natural substance on Earth?",
    "In what city is the Eiffel Tower located?",
    "What is the largest organ in the human body?",
    "Who wrote the play 'Romeo and Juliet'?",
    ]

    contexts = [
    "The largest planet in our solar system is Jupiter. It is known for its Great Red Spot and has many moons.",
    "The novel 'To Kill a Mockingbird' was written by Harper Lee.",
    "The capital of France is Paris.",
    "There are seven continents on Earth: Africa, Antarctica, Asia, Europe, North America, Australia, and South America.",
    "The boiling point of water is 100 degrees Celsius at standard atmospheric pressure.",
    "The Mona Lisa was painted by Leonardo da Vinci.",
    "The chemical symbol for gold is Au.",
    "The Titanic sank in 1912 after hitting an iceberg.",
    "Albert Einstein developed the theory of relativity.",
    "The smallest prime number is 2.",
    "The largest ocean on Earth is the Pacific Ocean.",
    "George Washington was the first president of the United States.",
    "The longest river in the world is the Nile River.",
    "Mars is known as the Red Planet due to its reddish appearance.",
    "The element with atomic number 1 is hydrogen.",
    "The lion is often referred to as the king of the jungle.",
    "The hardest natural substance on Earth is diamond.",
    "The Eiffel Tower is located in Paris, France.",
    "The largest organ in the human body is the skin.",
    "The play 'Romeo and Juliet' was written by William Shakespeare.",
    ]


    assert len(questions) == len(contexts), "Questions and contexts must be of the same length"

    answers = process_questions(questions, contexts)

    for question, answer in zip(questions, answers):
        print(f"Question: {question}")
        print(f"Answer: {answer}")
