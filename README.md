
# RAG Project with Roberta for Question Answering

This project demonstrates a basic Retrieval-Augmented Generation (RAG) approach using the `deepset/roberta-base-squad2` model from Hugging Face's Transformers library. The goal is to answer a set of predefined questions using given context data.

## Model and Libraries
We use `RobertaTokenizer` and `RobertaForQuestionAnswering` from the `transformers` library to tokenize the questions and contexts and predict the answer spans.

### Key Components:
- **Model**: `deepset/roberta-base-squad2` - A fine-tuned version of Roberta for question-answering tasks.
- **Libraries**: 
  - `transformers` for model and tokenizer
  - `torch` for tensor manipulation and model inference

## Project Files
- **Main Code**: Contains the code to process a list of questions and their corresponding contexts to generate answers.
- **Predefined Questions and Contexts**: A set of general knowledge questions and associated contexts.

## Installation
To get started, ensure you have the following libraries installed:

```bash
pip install torch transformers
```

## How to Run

1. **Set up the model**:  
   The code loads the Roberta model and tokenizer to perform question answering. The function `answer_question` takes a question and context as input and returns the predicted answer.

2. **Process questions and contexts**:  
   The `process_questions` function handles multiple question-context pairs and returns answers for each.

3. **Run the script**:
   The main part of the script contains a list of predefined questions and corresponding contexts. Simply run the script to see the answers:

   ```bash
   python <script_name>.py
   ```

   Example output:

   ```bash
   Question: What is the largest planet in our solar system?
   Answer: Jupiter
   ```

## Code Explanation

- **`answer_question(question, context)`**: This function takes a question and context, tokenizes them, and uses the Roberta model to predict the start and end indices of the answer span within the context.
  
- **`process_questions(questions, contexts)`**: This function iterates through a list of questions and contexts, calling `answer_question` for each pair and returning the list of answers.

The code contains comments to explain each function, making it easy to understand and modify.

## Notes
- This project is a demonstration of RAG using a simple question-answering model. It works well for fact-based questions with provided contexts.
- The predefined questions and contexts are simple, but you can extend the project by integrating more complex retrieval mechanisms to dynamically fetch relevant contexts.
