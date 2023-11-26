from transformers import AutoTokenizer
from openai import AsyncOpenAI
import json
import asyncio
from pprint import pprint

client = AsyncOpenAI(api_key="")


def calculate_tokens(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    tokens_len = len(tokens)
    return tokens_len


def get_json(text):
    parsed_json = None
    if "{" in text:
        try:
            parsed_json = json.loads(text)
        except Exception as ex:
            print(ex)
    return parsed_json


async def gpt_create_text():
    prompt = f"Write a very long unique text on any topic, be verbose and explain on detail"
    completion = await client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=3000
    )

    return ' '.join(completion.choices[0].text.split())


async def gpt_create_closer(text, change_percentage):
    print(change_percentage)

    if change_percentage < 0:
        change_vector = "less verbose"
    else:
        change_vector = "more verbose"

    abs_change_percentage = abs(change_percentage)

    if abs_change_percentage < 25:
        verbosity_degree = "barely"
    elif 25 <= abs_change_percentage < 55:
        verbosity_degree = "marginally"
    elif 55 <= abs_change_percentage < 85:
        verbosity_degree = "slightly"
    else:
        verbosity_degree = "A bit"

    print(verbosity_degree, change_vector)

    prompt = f"Make the following text {verbosity_degree} {change_vector}:\n" + text
    completion = await client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=3000
    )

    return ' '.join(completion.choices[0].text.split())


async def gpt_create_questions(text):
    prompt = f"""
Write unique questions and answers for the following text:

{text}

The questions should be straightforward enough for BERT model and with a clear answer like those in the squad dataset, and the answers should consist of either a single word or a pair of words. Return this information in JSON format structured as follows:
{{
    "questions": [{{"question": "question 1", "answers": ["answer 1", "answer 2"]}}, {{"question": "question 2", "answers": ["answer 1", "answer 2"]}}]
}}
Ensure that the "answers" field contains a list of acceptable answers corresponding to the generated question.
"""
    completion = await client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=900
    )
    answer = ' '.join(completion.choices[0].text.split())

    return get_json(answer)


async def gpt_create_example(min_tk_n, max_tk_n):
    average_q_len = 15  # Average question length in tokens

    # Generate a text
    text = await gpt_create_text()
    print(text)

    # Calculate the number of tokens in the text
    tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
    tokens_len = calculate_tokens(tokenizer, text)

    i = 0
    while not min_tk_n <= tokens_len + average_q_len < max_tk_n and i < 10:
        change_percentage = ((int((min_tk_n + max_tk_n)/2) - (tokens_len + average_q_len))/(tokens_len + average_q_len))*100

        text = await gpt_create_closer(text, change_percentage)

        tokens_len = calculate_tokens(tokenizer, text)
        print(i, tokens_len + average_q_len, text)
        i += 1

    if not min_tk_n <= tokens_len + average_q_len < max_tk_n:
        return

    questions = None
    k = 0
    while not questions and k < 3:
        questions = await gpt_create_questions(text)
        print(questions)

    if not questions:
        return

    pprint(questions)

    for question_answer in questions['questions']:
        example = {
            "context": text,
            "question": question_answer["question"],
            "answers": question_answer["answers"],
            "context_tokens_len": tokens_len,
            "question_tokens_len": calculate_tokens(tokenizer, question_answer["question"])
        }

        add_example_to_file(example, 'dataset.jsonl')


def add_example_to_file(example, file_name):
    with open(file_name, 'a') as f:
        f.write(json.dumps(example) + '\n')


async def main():
    for _ in range(5):
        min_tk_n = 400
        max_tk_n = 512
        await gpt_create_example(min_tk_n, max_tk_n)
    for _ in range(5):
        min_tk_n = 650
        max_tk_n = 750
        await gpt_create_example(min_tk_n, max_tk_n)
    for _ in range(5):
        min_tk_n = 900
        max_tk_n = 1024
        await gpt_create_example(min_tk_n, max_tk_n)
    for _ in range(5):
        min_tk_n = 1400
        max_tk_n = 1500
        await gpt_create_example(min_tk_n, max_tk_n)


if __name__ == '__main__':
    asyncio.run(main())
