import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import json
import codecs
import jsonlines
from faithscore.framework import FaithScore

OUTPUT_DIR = '/mnt/data/yuxi/dpo_llava/outputs/predictions/faithscore'


json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def load_jsonline(fname):
    with jsonlines.open(fname, mode='r') as reader:
        data = [r for r in reader]
    return data


def extract_outputs(fname, rawfile):
    raw_qas = {x['question_id']:x for x in load_jsonline(rawfile)}
    predictions = load_jsonline(fname)
    images, answers, info = [], [], []
    for pred in predictions:
        sample = raw_qas[pred['question_id']]
        info.append({
            'question': sample['text'], 
            'output': pred.get('output', pred['text']), 
            'type': sample.get('type', 'caption'),
        })
        images.append(f'/mnt/data/yuxi/coco/coco2014/val2014/COCO_val2014_{pred["question_id"]}.jpg')
        answers.append(pred['text'])
    return images, answers, info


def cal_faithscore(fname, scorer):
    rawfile = 'coco2014_sample_1k.jsonl' if 'caption-' in fname else 'test_qa_1000x3.jsonl'
    images, answers, info = extract_outputs(fname, f'/home/yuxi/Projects/LLaVA-DPO/playground/data/faithscore/{rawfile}')
    score, sentence_score, scores_list = scorer.faithscore(answers, images)
    instance_score, sentence_level_score = scores_list
    if 'caption-' in fname:
        scores = {
            'instance_score': {'avg': score, 'all': instance_score},
            'sentence_level_score': {'avg': sentence_score, 'all': sentence_level_score},
        }
    else:
        scores = {
            'instance_score': {
                'avg': score, 
                'complex': sum(instance_score[:1000]) / 1000, 
                'detail': sum(instance_score[1000:2000]) / 1000, 
                'conv': sum(instance_score[2000:]) / 1000, 
                'all': instance_score,
            },
            'sentence_level_score': {
                'avg': sentence_score, 
                'complex': sum(sentence_level_score[:1000]) / 1000, 
                'detail': sum(sentence_level_score[1000:2000]) / 1000, 
                'conv': sum(sentence_level_score[2000:]) / 1000, 
                'all': sentence_level_score,
            },
        }
    for k, v in scores.items():
        print(f'===== {k} =====')
        for kk, vv in v.items():
            if kk != 'all': continue
            print(kk, vv)
    json_dump(scores, fname.replace('.jsonl', '_scores.json'))


if __name__ == '__main__':
    scorer = FaithScore(vem_type='llava', 
                        api_key='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', 
                        llava_path='liuhaotian/llava-v1.5-13b', 
                        use_llama=False)
    
    # cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-qa-answer-file-baseline.jsonl'), scorer)
    # cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-qa-answer-file-ours.jsonl'), scorer)
    # cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-qa-answer-file-sherlock-easymix.jsonl'), scorer)
    # cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-qa-answer-file-sherlock.jsonl'), scorer)
    
    # cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-caption-answer-file-baseline.jsonl'), scorer)
    # cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-caption-answer-file-ours.jsonl'), scorer)
    # cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-caption-answer-file-sherlock.jsonl'), scorer)
    # cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-caption-answer-file-sherlock-easymix.jsonl'), scorer)

    cal_faithscore(os.path.join(OUTPUT_DIR, 'caption-baseline.jsonl'), scorer)
    cal_faithscore(os.path.join(OUTPUT_DIR, 'qa-baseline.jsonl'), scorer)
    