import os
import time
import json
import codecs
import jsonlines
from tqdm import tqdm
from openai import OpenAI

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))

def json_dump(x, p):
    with open(p, 'w') as f:
        json.dump(x, f, indent=2)

def jsonlines_load(fname):
    with jsonlines.open(fname, mode='r') as reader:
        data = [r for r in reader]
    return data

def jsonlines_dump(x, p):
    with jsonlines.open(p, mode='w') as writer:
        writer.write_all(x)

def call_openai(client: OpenAI, prompt):    
    response = None
    while response is None:
        try:
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
        except Exception as e:
            print(e)
            print('retrying...')
            time.sleep(10)
            continue
    return response.choices[0].message.content

def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def YOrN_match_prompt(question, answer):
    tmpl = (
        "You are an AI assistant who will help me to match an answer with two options of a question. "
        "The options are only Yes / No. "
        "You are provided with a question and an answer, and you need to find which option (Yes / No) is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Unknown. "\
        "Your should output a single word among the following 3 choices: Yes, No, Unknown.\n"
        "Example 1: \n"
        "Question: Is the word in this image 'Hello'?\nAnswer: The word in this image is 'Hello'.\nYour output: Yes\n"
        "Example 2: \n"
        "Question: Is the word in this image 'Hello'?\nAnswer: The word in this image is not 'Hello'.\nYour output: No\n"
        # "Example 3: \n"
        # "Question: Is there direct contact between the cup and wall?\nAnswer: The man is holding the cup in his hand while standing next to the wall.\nYour output: No\n"
        # "Example 4: \n"
        # "Question: Is there direct contact between the man and cup?\nAnswer: The man is holding the cup, and it is positioned in front of him, but he is not physically touching it.\nYour output: Yes\n"
        "Example 3: \n"
        "Question: {}\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, answer)

def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'

def process_output(fname):
    client = OpenAI(api_key='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')
    save_internal = 200
    
    output_path = '.'.join(fname.split('.')[:-1]) + '_response.json'    
    outputdata = jsonlines_load(fname)
    data = []
    if os.path.exists(output_path):
        data = json_load(output_path)
    for i, dt in enumerate(tqdm(outputdata)):
        if i < len(data): continue
        if dt['question_id'] < 1005:
            data.append({
                'id': dt['question_id'],
                'response': dt['text'],
            })
        else:
            text = dt['text']
            prompt = YOrN_match_prompt(dt['prompt'], text)
            extracted = text.split('.')[0].strip().split(',')[0].strip()
            data.append({
                'id': dt['question_id'],
                'response': extracted if extracted.lower() in ['yes', 'no'] else YOrN_Extraction(call_openai(client, prompt)),
            })
        if len(data) % save_internal == 0:
            json_dump(data, output_path)
    json_dump(data, output_path)


if __name__ == '__main__':
    process_output('/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/amber/baseline.jsonl')
    # process_output('/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/amber/nolb-llava-cdpo-wl-e4.jsonl')
    # process_output('/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/amber/nolb-llava-cdpo-lbwl-e4.jsonl')
    # process_output('/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/amber/nolb-llava-cdpo-lbwl-sl-e4.jsonl')
    # process_output('/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/amber/nolb-llava-cdpo-dylbwl-e4.jsonl')
    # process_output('/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/amber/nolb-llava-cdpo-dylbwl-sl-e4.jsonl')
