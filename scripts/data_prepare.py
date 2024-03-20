import os
import json
import codecs
import random
import jsonlines
from tqdm import tqdm
from copy import deepcopy

from transformers import AutoTokenizer

random.seed(0)

json_load = lambda x: json.load(codecs.open(x, mode='r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)

def jsonlines_load(x):
    with jsonlines.open(x, mode='r') as reader:
        data = [r for r in reader]
    return data

def jsonlines_dump(x, p):
    with jsonlines.open(p, mode='w') as writer:
        writer.write_all(x)

data_path = '/mnt/data/yuxi/dpo_llava/data/similarityscores_mix-duplicate-reduce_27k.json'
data_path2 = '/mnt/data/yuxi/dpo_llava/data/sherlock_filtered_all/sherlock_similarityscores_mix-duplicate-reduce_187k.json'
image_folder = '/mnt/data'
ptx_data_path = '/mnt/data/yuxi/llava/data/llava_v1_5_mix665k.json'
ptx_image_folder = '/mnt/data/yuxi'

def cat_instructions(raw, tokenizer:AutoTokenizer):
    inst_dict = {}
    for dt in tqdm(raw):
        key = dt['image'].split('/')[0].strip() if 'image' in dt else 'gpt'
        if key != 'gpt' and not os.path.exists(os.path.join(ptx_image_folder, dt['image'])): continue
        raw_text = '\n'.join([x['value'] for x in dt['conversations']])
        if len(tokenizer.encode(raw_text)) > 2000: continue
        if key not in inst_dict: inst_dict[key] = {}
        _id = dt['id']
        if _id not in inst_dict[key]: inst_dict[key][_id] = []
        dt.update({'category': 'instruct'})
        inst_dict[key][_id].append(dt)
    return inst_dict

def cat_preferences(raw):
    pref_dict = {'coco': {}, 'vcr': {}, 'vg': {}}
    for dt in tqdm(raw):
        key = 'coco' if 'COCO' in dt['id'] else 'vcr'
        key = 'vg' if 'VG_100K' in dt['id'] else key
        if key == 'coco':
            _id = dt['id'].split('__')[0].split('_')[-1].strip()
        elif key == 'vg':
            _id = '/'.join(dt['id'].split('__')[:2])
        elif '/' in dt['id']:
            _id = dt['id'].split('__')[0]
        else:
            _id = '/'.join(dt['id'].split('__')[:2])
        if _id not in pref_dict[key]: pref_dict[key][_id] = []
        images = dt.get('image', dt.get('images', []))
        if isinstance(images, str): images = [images]
        for i, img in enumerate(images):
            if not os.path.exists(os.path.join(image_folder, img)):
                images[i] = f'yuxi/sherlock/img2img/train/{img}'
        if 'image' in dt:
            dt['image'] = images[0]
        else:
            dt['images'] = images
        dt.update({'category': 'region' if '/img2img/' in images[0] else 'vqa'})
        pref_dict[key][_id].append(dt)
    return pref_dict

def collect_data(inst_dict, pref_dict):
    data = {}
    for key, value in pref_dict.items():
        for _id, samples in value.items():
            idx = (key, _id)
            if idx not in data: data[idx] = []
            data[idx] += samples
    for key, value in inst_dict.items():
        for _id, samples in value.items():
            idx = (key, _id)
            if idx not in data: data[idx] = []
            data[idx] += samples
    stat = {x: [] for x in ['instruct', 'region', 'vqa']}
    cnt = []
    for idx, samples in tqdm(data.items()):
        candidates = [x for x in samples if x['category'] != 'instruct']
        cnt.append(len(candidates))
        if len(candidates):
            for sample in candidates:
                if sample['category'] == 'vqa' or random.random() < 6 / len(candidates):
                    stat[sample['category']].append(sample)
        else:
            qa_dict = {}
            for sample in samples:
                for i in range(len(sample['conversations']) // 2):
                    qu, ans = sample['conversations'][i * 2]['value'], sample['conversations'][i * 2 + 1]['value']
                    if not qu.strip() or not ans.strip(): continue
                    qa_dict[qu] = ans
            if len(qa_dict) < 2: continue
            candidates = [x for x in samples if all(x['conversations'][i * 2]['value'] in qa_dict for i in range(len(x['conversations']) // 2))]
            if not len(candidates): continue
            sample = random.choice(candidates)
            contrastive_conversations = deepcopy(sample['conversations'])
            qu = random.choice([q for q in qa_dict if q != contrastive_conversations[-2]['value']])
            contrastive_conversations[-1]['value'] = qa_dict[qu]
            sample['contrastive_conversations'] = contrastive_conversations
            if 'image' in sample:
                sample.update({'image': f'yuxi/{sample["image"]}'})
            stat['instruct'].append(sample)
    data = []
    ratios = [27393/168909, 27393/121072, 0.9]
    for (k, v), ratio in zip(stat.items(), ratios):
        print(k, len(v))
        cnt = 0
        for vv in v:
            if random.random() > ratio: continue
            data.append(vv)
            cnt += 1
        print(cnt)
    json_dump(data, '/mnt/data/yuxi/dpo_llava/data/all_preferences_{}k.json'.format(len(data) // 1000))

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        "liuhaotian/llava-v1.5-7b",
        model_max_length=2048,
        padding_side='left',
        trust_remote_code=True,
        use_fast=False,
        use_auth_token="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",
    )
    
    instructions = json_load(ptx_data_path)
    inst_dict = cat_instructions(instructions, tokenizer)    
    preferences = json_load(data_path) + json_load(data_path2)
    pref_dict = cat_preferences(preferences)

    collect_data(inst_dict, pref_dict)
    