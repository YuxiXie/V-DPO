import os
import time
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

# data_path = '/mnt/data/yuxi/dpo_llava/data/similarityscores_mix-duplicate-reduce_27k.json'
data_path = '/mnt/data/yuxi/dpo_llava/data/similarityscores_mix-duplicate-reduce_27k_v4.json'
# data_path2 = '/mnt/data/yuxi/dpo_llava/data/sherlock_filtered_all/sherlock_similarityscores_mix-duplicate-reduce_187k.json'
data_path2 = '/mnt/data/yuxi/dpo_llava/data/sherlock_filtered_all/sherlock_similarityscores_mix-duplicate-reduce_171k_v3.json'
# data_path2 = '/mnt/data/yuxi/dpo_llava/data/sherlock_filtered_all/sherlock_similarityscores_mix-duplicate-reduce_171k_v3_cropped.json'
image_folder = '/mnt/data'
ptx_data_path = '/mnt/data/yuxi/llava/data/llava_v1_5_mix665k.json'
ptx_image_folder = '/mnt/data/yuxi'

instruct_path = '/mnt/data/yuxi/dpo_llava/data/instructions_v3.json'
inst_pref_path = '/mnt/data/yuxi/dpo_llava/data/instructions_and_preferences_v4_cropped.json'

def cat_instructions(raw, tokenizer:AutoTokenizer):
    if os.path.exists(instruct_path):
        _dict = json_load(instruct_path)
        return _dict['inst_dict'], _dict['qa_dict']
    inst_dict, qa_dict = {}, {'single word or phrase': {}}
    for dt in tqdm(raw):
        key = dt['image'].split('/')[0].strip() if 'image' in dt else 'gpt'
        if key != 'gpt' and not os.path.exists(os.path.join(ptx_image_folder, dt['image'])): continue
        raw_text = '\n'.join([x['value'] for x in dt['conversations']])
        if len(tokenizer.encode(raw_text)) > 1000: continue
        if key not in inst_dict: inst_dict[key] = {}
        _id = dt['id']
        if _id not in inst_dict[key]: inst_dict[key][_id] = []
        dt.update({'category': 'instruct'})
        inst_dict[key][_id].append(dt)
        flag = dt['conversations'][0]['value'].count('nswer the question using a single word or phrase.')
        for i in range(len(dt['conversations']) // 2):
            qu = dt['conversations'][i * 2]['value'].lower().replace('<image>', '').replace('answer the question using a single word or phrase.', '').strip()
            ans = dt['conversations'][i * 2 + 1]['value'].lower().strip()
            if flag:
                if qu not in qa_dict['single word or phrase']: qa_dict['single word or phrase'][qu] = {}
                if dt['id'] not in qa_dict['single word or phrase'][qu] or random.random() < .5:
                    qa_dict['single word or phrase'][qu][dt['id']] = ans
            else:
                if qu not in qa_dict: qa_dict[qu] = {}
                # if ans not in qa_dict[qu]: qa_dict[qu].append(ans)
                if dt['id'] not in qa_dict[qu] or random.random() < .5:
                    qa_dict[qu][dt['id']] = ans
    json_dump({'inst_dict': inst_dict, 'qa_dict': qa_dict}, instruct_path)
    return inst_dict, qa_dict

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

def collect_data(inst_dict, pref_dict, qa_dict):
    answer_with_length = {'short': {}, 'normal': {}, 'middle': {}, 'long': {}, 'bbox': {}, 'region': {}}
    for qu, ans_dict in qa_dict.items():
        if qu == 'single word or phrase':
            for adict in ans_dict.values():
                answer_with_length['short'].update(adict)
        elif qu.count('provide the bounding box coordinate'):
            answer_with_length['bbox'].update(ans_dict)
        elif qu.count('a short description for this region'):
            answer_with_length['region'].update(ans_dict)
        else:
            for _id, answer in ans_dict.items():
                _len = len(answer.split())
                if _len > 128:
                    answer_with_length['long'][_id] = answer
                elif _len > 32:
                    answer_with_length['middle'][_id] = answer
                else:
                    answer_with_length['normal'][_id] = answer
    
    if False and os.path.exists(inst_pref_path):
        stat = json_load(inst_pref_path)
    else:
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
            elif random.random() < .15:
                questions, answers = [], []
                for sample in samples:
                    for i in range(len(sample['conversations']) // 2):
                        questions.append(sample['conversations'][i * 2]['value'].lower().replace('<image>', '').replace('answer the question using a single word or phrase.', '').strip())
                        answers.append(sample['conversations'][i * 2 + 1]['value'].lower().strip())
                sample = random.choice(samples)
                flag = 'Answer the question using a single word or phrase.' in sample['conversations'][0]['value']
                contrastive_conversations = deepcopy(sample['conversations'])
                qu = contrastive_conversations[-2]['value'].lower().replace('<image>', '').replace('answer the question using a single word or phrase.', '').strip()
                ans = contrastive_conversations[-1]['value'].lower()
                if ans == 'yes':
                    candidates = ['No']
                elif ans == 'no':
                    candidates = ['Yes']
                elif ans in ['a', 'b', 'c', 'd']:
                    candidates = [x for x in 'ABCD' if x.lower() != ans]
                elif qu.count('provide the bounding box coordinate'):
                    candidates = answer_with_length['bbox']
                elif qu.count('a short description for this region'):
                    candidates = answer_with_length['region']
                elif flag:
                    candidates = qa_dict['single word or phrase'].get(qu, {})
                else:
                    candidates = qa_dict.get(qu, {})
                candidates = [v for k, v in candidates.items() if k != sample['id'] and v != contrastive_conversations[-1]['value'].lower()] if isinstance(candidates, dict) else candidates
                if not len(candidates):
                    _len = len(contrastive_conversations[-1]['value'].split())
                    if flag:
                        candidates = [v for k, v in answer_with_length['short'].items() if k != sample['id']]
                    elif _len > 128:
                        candidates = [v for k, v in answer_with_length['long'].items() if k != sample['id']]
                    elif _len > 32:
                        candidates = [v for k, v in answer_with_length['middle'].items() if k != sample['id']]
                    else:
                        candidates = [v for k, v in answer_with_length['normal'].items() if k != sample['id'] and v != contrastive_conversations[-1]['value'].lower()]
                contra_ans = random.choice(candidates)
                contrastive_conversations[-1]['value'] =  contra_ans[0].upper() + contra_ans[1:]
                sample['contrastive_conversations'] = contrastive_conversations
                if 'image' in sample:
                    sample.update({'image': f'yuxi/{sample["image"]}'})
                stat['instruct'].append(sample)
                # print(sample)
        json_dump(stat, inst_pref_path)
    data = []
    # ratios = [27393/201286, 27393/121072, 0.9]
    ratios = [21915/30360, 21915/117747, 0.8]
    for (k, v), ratio in zip(stat.items(), ratios):
        print(k, len(v))
        cnt = 0
        for vv in v:
            if random.random() > ratio: continue
            data.append(vv)
            cnt += 1
        print(cnt)
    json_dump(data, '/mnt/data/yuxi/dpo_llava/data/all_preferences_easy_{}k_v4.json'.format(len(data) // 1000))

def split_data(fname):
    rawdata = json_load(fname)
    instructions = []
    vcr_txt_pair, coco_txt_pair, sherlock_txt_pair = [], [], []
    vcr_img_pair, coco_img_pair, sherlock_img_pair, sherlock_img_pair_inf = [], [], [], []
    for dt in tqdm(rawdata):
        if dt['category'] == 'instruct':
            if 'image' in dt:
                instructions.append(dt)
        elif dt['category'] == 'vqa':
            img = dt['images'][0] if 'images' in dt else dt['image']
            if '/coco/' in img:
                if 'images' in dt:
                    coco_img_pair.append(dt)
                else:
                    coco_txt_pair.append(dt)
            elif 'images' in dt:
                vcr_img_pair.append(dt)
            else:
                vcr_txt_pair.append(dt)
        else:
            if 'image' in dt:
                sherlock_txt_pair.append(dt)
            elif 'suggested or indicated' in dt['conversations'][0]['value']:
                sherlock_img_pair_inf.append(dt)
            else:
                sherlock_img_pair.append(dt)
    
    datadict = {
        'instructions': random.sample(instructions, 1000),
        'vcr_txt_pair': random.sample(vcr_txt_pair, 1000),
        'coco_txt_pair': random.sample(coco_txt_pair, 1000),
        'sherlock_txt_pair': random.sample(sherlock_txt_pair, 1000),
        'vcr_img_pair': random.sample(vcr_img_pair, 1000),
        'coco_img_pair': random.sample(coco_img_pair, 1000),
        'sherlock_img_pair': random.sample(sherlock_img_pair, 1000),
        'sherlock_img_pair_inf': random.sample(sherlock_img_pair_inf, 1000),
    }
    data = []
    for k, v in datadict.items():
        for x in v:
            x.update({'category': k})
            data.append(x)
    json_dump(data, fname.replace('.json', '_foreval.json'))

def visualize_eval_rst(fname, tokenizer:AutoTokenizer):
    import torch
    from llava_dpo.utils import get_log_probs
    
    id2txt = {
        0: 'instructions', 
        -1: 'vcr_txt_pair', -2: 'coco_txt_pair', -3: 'sherlock_txt_pair',
        -4: 'vcr_img_pair', -5: 'coco_img_pair', -6: 'sherlock_img_pair', 
        -7: 'sherlock_img_pair_inf',
    }
        
    def _get_prob(_ids, _labels, _logprobs):
        lp, _len = get_log_probs(torch.LongTensor(_ids), torch.BoolTensor(_labels), torch.Tensor(_logprobs), 
                                 is_answer=True, final_answer=True, return_len=True)
        # prob = torch.exp(lp / _len).item()
        prob = torch.exp(lp).item()
        return (prob, _len.item())
    
    def _get_text(_ids):
        _ids = _ids[:]
        img_idx = _ids.index(-200)
        _ids[img_idx-1:img_idx+2] = [529, 3027, 29958]
        text = tokenizer.decode(_ids)
        question = 'ASSISTANT: '.join(text.split('ASSISTANT: ')[:-1]).replace('<s> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.', '').strip()
        answer = text.split('ASSISTANT: ')[-1]
        return question, answer
    
    outputs = jsonlines_load(fname)
    results = []
    for out in tqdm(outputs[:3310]):
        category_id = out['category_id']
        better_ids, worse_ids = out['better_ids'], out['worse_ids']
        better_labels, worse_labels = out['better_labels'], out['worse_labels']
        better_logprobs, worse_logprobs = out['better_logprobs'], out['worse_logprobs']
        better_randimg_logprobs, worse_randimg_logprobs = out['better_randimg_logprobs'], out['worse_randimg_logprobs']
        question, better_ans = _get_text(better_ids)
        _, worse_ans = _get_text(worse_ids)
        results.append({
            'category': id2txt[int(category_id)],
            'question': question,
            'better_answer': better_ans,
            'better': _get_prob(better_ids, better_labels, better_logprobs),
            'better_randimg': _get_prob(better_ids, better_labels, better_randimg_logprobs),
            'worse_answer': worse_ans,
            'worse': _get_prob(worse_ids, worse_labels, worse_logprobs),
            'worse_randimg': _get_prob(worse_ids, worse_labels, worse_randimg_logprobs),
        })
    
    better_sent_prob = {v:[] for v in id2txt.values()}
    worse_sent_prob = {v:[] for v in id2txt.values()}
    better_rand_sent_prob = {v:[] for v in id2txt.values()}
    worse_rand_sent_prob = {v:[] for v in id2txt.values()}
    better_worse = {v:[] for v in id2txt.values()}
    better_rand = {v:[] for v in id2txt.values()}
    rand_worse = {v:[] for v in id2txt.values()}
    for rst in results:
        better_sent_prob[rst['category']].append(rst['better'][0] ** (1/rst['better'][1]))
        worse_sent_prob[rst['category']].append(rst['worse'][0] ** (1/rst['worse'][1]))
        better_rand_sent_prob[rst['category']].append(rst['better_randimg'][0] ** (1/rst['better_randimg'][1]))
        worse_rand_sent_prob[rst['category']].append(rst['worse_randimg'][0] ** (1/rst['worse_randimg'][1]))
        better_worse[rst['category']].append(better_sent_prob[rst['category']][-1] - worse_sent_prob[rst['category']][-1])
        better_rand[rst['category']].append(better_sent_prob[rst['category']][-1] - better_rand_sent_prob[rst['category']][-1])
        rand_worse[rst['category']].append(worse_rand_sent_prob[rst['category']][-1] - worse_sent_prob[rst['category']][-1])
    
    for x in [
        better_sent_prob, worse_sent_prob, 
        better_rand_sent_prob, worse_rand_sent_prob, 
        better_worse, better_rand, rand_worse
    ]:
        print('==============')
        for k, v in x.items():
            print(k, sum(v)/max(1,len(v)))
    
    # import ipdb; ipdb.set_trace()

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
    inst_dict, qa_dict = cat_instructions(instructions, tokenizer)
    # import ipdb; ipdb.set_trace()
    preferences = json_load(data_path) + json_load(data_path2)
    pref_dict = cat_preferences(preferences)
    
    collect_data(inst_dict, pref_dict, qa_dict)
    
    # split_data('/mnt/data/yuxi/dpo_llava/data/all_preferences_easy_74k_v2.json')
    # visualize_eval_rst('/mnt/data/yuxi/dpo_llava/outputs/checkpoints/llava-triple-eval/eval_result.jsonl',
    #                    tokenizer)
    # visualize_eval_rst('/mnt/data/yuxi/dpo_llava/outputs/checkpoints/llava-triple-eval/eval_result_trained.jsonl',
    #                    tokenizer)
    