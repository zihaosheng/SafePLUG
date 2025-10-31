import argparse
import os, json, ast, re, time

import openai
from tqdm import tqdm

NUM_SECONDS_TO_SLEEP = 0.5

def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    # print('success!')
    return response['choices'][0]['message']['content']

def replace_quotes(match):
    key = match.group(1)
    value = match.group(3).replace('"', "'").replace("\\\'", "'")
    return f'{key}{value}{match.group(4)}'

def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        invcomma_pattern = r'("(score|explanation)": *")(.*?)(", *"|" *})'
        invcomma_pattern_ans = r'("(explanation)": *")(.*?)(" *})'
        review = review[review.find('{'): len(review)-review[::-1].find('}')]
        review = re.sub(invcomma_pattern, replace_quotes, review)
        review = re.sub(invcomma_pattern_ans, replace_quotes, review)
        
        review_dict = ast.literal_eval(review)
        score = review_dict.get("score", 0)
        explanation = review_dict.get("explanation", 0)
        return int(score), explanation
    except SyntaxError as e:
        print(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return 0, ''
    except ValueError as e:
        print(f"Value error parsing the review string: {e}. Review content: {review}")
        return 0, ''
    except Exception as e:
        print(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return 0, ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-a', '--answer')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    f_ans = open(os.path.expanduser(args.answer))
    args.output = args.answer.replace('.jsonl', '_review.json')
    print("Answer file: ", args.answer)
    print("Output file: ", args.output)

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    js_list = []
    reviews = []
    score_list = []
    idx = 0
    for ans_js in tqdm(f_ans, desc='Evaluating'):
        # if idx == 100:
        #     break
        ans = json.loads(ans_js)
        question = ans['prompt'].replace('<im_end>', '').replace('<im_patch>', '')
        gt = ans['gt']
        answer = ans['text']

        prompts = ( "Evaluate the following question-answer pair:\\n"
                            f"Question: {question}\\n"
                            f"Correct Answer: {gt}\\n"
                            f"Predicted Answer: {answer}\\n\\n"
                            "Rate the Predicted Answer based on the Correct Answer on a scale from 0 to 100, with higher scores indicating that the Predicted Answer is closer to the Correct Answer. Your rating should be accurate to single digits like 62, 78, 41, etc."
                            '\nYour rating should consider the reasonableness, detail, and consistency.'
                            '\nPlease generate the response in the form of a Python dictionary string with keys "score", where its value is in INTEGER, not STRING, and "explanation" giving short and concise reasoning behind the score.'
                            '\nFor example, your response should look like this: {"score": 45, "explanation": "..."}')


        if idx >= len(cur_reviews):
            review = get_eval(prompts, args.max_tokens)
            reviews.append(review)

            # To avoid the rate limit set by OpenAI
            time.sleep(NUM_SECONDS_TO_SLEEP)

            score, explanation = parse_score(review)
            score_list.append(score)
            cur_js = {
                'id': idx+1,
                'score': score,
                'explanation': explanation,
                'question_id': ans['question_id'],
                'gt': gt,
                'answer': answer,
                'review': review
                }
                
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1

    print(f'Average score: {sum(score_list) / len(score_list)}')
    review_file.close()