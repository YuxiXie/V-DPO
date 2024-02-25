import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import jsonlines
from faithscore.framework import FaithScore

OUTPUT_DIR = '/mnt/data/yuxi/dpo_llava/outputs/predictions'


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
            'output': sample['output'],
            'type': sample['type'],
        })
        images.append(f'/mnt/data/yuxi/coco/coco2014/val2014/COCO_val2014_{pred["question_id"]}.jpg')
        answers.append(pred['text'])
    return images, answers, info


def cal_faithscore(fname, scorer):
    rawfile = 'coco2014_sample_1k.jsonl' if '-caption-' in fname else 'test_qa_1000x3.jsonl'
    images, answers, info = extract_outputs(fname, f'/home/yuxi/Projects/LLaVA-DPO/playground/data/faithscore/{rawfile}')
    try:
        score, sentence_score, scores_list = scorer.faithscore(answers, images)
    except:
        import ipdb; ipdb.set_trace()
    import ipdb; ipdb.set_trace()
    
    
if __name__ == '__main__':
    scorer = FaithScore(vem_type='llava', api_key='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', 
                        llava_path='liuhaotian/llava-v1.5-13b', use_llama=False)
    
    cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-qa-answer-file-baseline.jsonl'), scorer)
    cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-qa-answer-file-ours.jsonl'), scorer)
    
    cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-caption-answer-file-baseline.jsonl'), scorer)
    cal_faithscore(os.path.join(OUTPUT_DIR, 'faithscore-caption-answer-file-ours.jsonl'), scorer)
