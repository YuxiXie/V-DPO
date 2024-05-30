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

data_path = '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/similarityscores_mix-duplicate-reduce_18k_v10.json'
data_path2 = '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/sherlock_filter/sherlock_similarityscores_mix-duplicate-reduce_22k_v10.json'
image_folder = '/home/users/nus/e0672129/scratch'
ptx_data_path = '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/llava_v1_5_mix665k.json'
ptx_image_folder = '/home/users/nus/e0672129/scratch/image_data'

instruct_path = '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/instructions_v5.json'
inst_pref_path = '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/instructions_and_preferences_v10-2.json'

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
                images[i] = img.replace('guanzhen/coco/Image/', 'image_data/coco/')
                images[i] = images[i].replace('yuxi/vcr/vcr_outputs/', 'image_data/vcr/vcr_outputs/')
                images[i] = images[i].replace('yuxi/vcr/vcr1images/', 'image_data/vcr/vcr1images/')
                if images[i] == img and '/' not in img:
                    images[i] = f'LLaVA-DPO/images/sherlock/train/{img}'
        if 'image' in dt:
            dt['image'] = images[0]
        else:
            dt['images'] = images
        dt.update({'category': 'region' if '/sherlock/' in images[0] else 'vqa'})
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
    
    if os.path.exists(inst_pref_path):
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
            elif random.random() < .5:
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
                    sample.update({'image': f'image_data/{sample["image"]}'})
                stat['instruct'].append(sample)
                # print(sample)
        json_dump(stat, inst_pref_path)
    data = []
    # ratios = [27393/201286, 27393/121072, 0.9]
    # ratios = [30000/36276, 21976/117747, 0.8]
    ratios = [1/20, 3/20, 1/5]
    yn_cnt = []
    for (k, v), ratio in zip(stat.items(), ratios):
        print(k, len(v))
        cnt = 0
        for vv in v:
            if random.random() > ratio: continue
            # elif vv['conversations'][-1]['value'].lower().startswith('yes') and random.random() > ratio * 2: continue
            # if vv['conversations'][-1]['value'].lower().startswith('no') and random.random() > 1/2: continue
            # if random.random() > ratio: continue
            if vv['conversations'][-1]['value'].lower().startswith('yes'):
                yn_cnt.append(2)
            elif vv['conversations'][-1]['value'].lower().startswith('no'):
                yn_cnt.append(-2)
            data.append(vv)
            cnt += 1
        print(cnt)
    print(yn_cnt.count(2), yn_cnt.count(-2))
    # import ipdb; ipdb.set_trace()
    json_dump(data, '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/all_preferences_{}k_v10.json'.format(len(data) // 1000))

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
    
    N = 500
    datadict = {
        'instructions': random.sample(instructions, N),
        'vcr_txt_pair': random.sample(vcr_txt_pair, N),
        'coco_txt_pair': random.sample(coco_txt_pair, N),
        'sherlock_txt_pair': random.sample(sherlock_txt_pair, N),
        'vcr_img_pair': random.sample(vcr_img_pair, N),
        'coco_img_pair': random.sample(coco_img_pair, N),
        'sherlock_img_pair': random.sample(sherlock_img_pair, N),
        'sherlock_img_pair_inf': random.sample(sherlock_img_pair_inf, N),
    }
    data = []
    for k, v in datadict.items():
        for x in v:
            x.update({'category': k})
            data.append(x)
    json_dump(data, '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/all_preferences_{}k_v10_foreval.json'.format(len(data) // 1000))

def visualize_eval_rst(fname, rawfname, tokenizer:AutoTokenizer):
    import math
    import torch
    import sys
    sys.path.append('/home/users/nus/e0672129/LLaVA-DPO')
    from llava_dpo.utils import get_log_probs
    
    id2txt = {
        0: 'instructions', 
        -1: 'vcr_txt_pair', -2: 'coco_txt_pair', -3: 'sherlock_txt_pair',
        -4: 'vcr_img_pair', -5: 'coco_img_pair', -6: 'sherlock_img_pair', 
        -7: 'sherlock_img_pair_inf',
    }
    
    def _get_prob(_ids, _labels, _logprobs):
        lp, _len = get_log_probs(torch.LongTensor(_ids), torch.BoolTensor(_labels), torch.Tensor(_logprobs), 
                                 label_mask=torch.BoolTensor(_labels),
                                 is_answer=True, final_answer=True, return_len=True)
        # prob = torch.exp(lp / _len).item()
        prob = torch.exp(lp).item()
        # return (prob, _len.item())
        return (lp, _len.item())
    
    def _get_text(_ids):
        _ids = _ids[:]
        img_idx = _ids.index(-200)
        _ids[img_idx-1:img_idx+2] = [529, 3027, 29958]
        text = tokenizer.decode(_ids)
        question = 'ASSISTANT: '.join(text.split('ASSISTANT: ')[:-1]).replace('<s> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.', '').strip()
        answer = text.split('ASSISTANT: ')[-1]
        return question, answer
    
    def _cal_prob(lp, cnt):
        return math.exp(lp / cnt)
    
    def _cal_ratio(lp1, cnt1, lp2, cnt2):
        return math.exp(lp1 / cnt1 - lp2 / cnt2)
    
    outputs = jsonlines_load(fname)
    results = []
    scores = []
    for out in tqdm(outputs):
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
        
        better_logit_diff = torch.Tensor(better_logprobs) - torch.Tensor(better_randimg_logprobs)
        better_selected_ids = better_logit_diff[:-1].ge(-1) * torch.LongTensor(better_ids[1:]).ge(-1) * torch.LongTensor(better_ids[1:]) * torch.BoolTensor(better_labels[:-1])
        # print(tokenizer.decode(better_selected_ids).replace('<unk>', '_'))
        
        worse_logit_diff = torch.Tensor(worse_logprobs) - torch.Tensor(worse_randimg_logprobs)
        worse_selected_ids = worse_logit_diff[:-1].ge(-1) * torch.LongTensor(worse_ids[1:]).ge(-1) * torch.LongTensor(worse_ids[1:]) * torch.BoolTensor(worse_labels[:-1])
        # print(tokenizer.decode(worse_selected_ids).replace('<unk>', '_'))
        
        scores.append({
            'question': question,
            'better_ids': better_ids[1:] + [tokenizer.eos_token_id],
            'better_logit_diff': better_logit_diff.tolist(),
            'better_label_mask': better_labels,
            'worse_ids': worse_ids[1:] + [tokenizer.eos_token_id],
            'worse_logit_diff': worse_logit_diff.tolist(),
            'worse_label_mask': worse_labels,
        })
    
    jsonlines_dump(scores, '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/eval_scores_0rimg.jsonl')
    
    raw_samples = json_load(rawfname)
    raw_samples_dict = {(x['conversations'][-2]['value'].replace('<image>', '').strip(), x.get('category', 'instructions'), len(x['conversations'])//2): x for x in raw_samples}
    
    better_sent_prob = {v:[] for v in id2txt.values()}
    worse_sent_prob = {v:[] for v in id2txt.values()}
    better_rand_sent_prob = {v:[] for v in id2txt.values()}
    worse_rand_sent_prob = {v:[] for v in id2txt.values()}
    better_worse = {v:[] for v in id2txt.values()}
    better_rand = {v:[] for v in id2txt.values()}
    rand_worse = {v:[] for v in id2txt.values()}
    clip_scores = {v:[] for v in id2txt.values()}
    for i, rst in enumerate(results):
        qu = rst['question'].split('USER:')[-1].replace('<image>', '').strip()
        rst['category'] = rst.get('category', 'instructions')
        raw_sample = raw_samples_dict.get((qu, rst['category'], rst['question'].count('USER:')), None)
        if raw_sample is None: continue
        
        if 'clip_scores' in raw_sample:
            clip_scores[rst['category']].append(raw_sample['clip_scores'][0] / (raw_sample['clip_scores'][0] + raw_sample['clip_scores'][1]))
        else:
            clip_scores[rst['category']].append(.8)
        
        better_sent_prob[rst['category']].append(_cal_prob(rst['better'][0], rst['better'][1]))
        worse_sent_prob[rst['category']].append(_cal_prob(rst['worse'][0], rst['worse'][1]))
        
        better_rand_sent_prob[rst['category']].append(_cal_prob(rst['better_randimg'][0], rst['better_randimg'][1]))
        worse_rand_sent_prob[rst['category']].append(_cal_prob(rst['worse_randimg'][0], rst['worse_randimg'][1]))
        
        better_worse[rst['category']].append(_cal_ratio(rst['better'][0], rst['better'][1], rst['worse'][0], rst['worse'][1]))
        better_rand[rst['category']].append(_cal_ratio(rst['better'][0], rst['better'][1], rst['better_randimg'][0], rst['better_randimg'][1]))
        rand_worse[rst['category']].append(_cal_ratio(rst['worse'][0], rst['worse'][1], rst['worse_randimg'][0], rst['worse_randimg'][1]))
        
        # if better_worse[rst['category']][-1] < 1 or better_rand[rst['category']][-1] < 1:
        #     import ipdb; ipdb.set_trace()
    
    for x in [
        better_sent_prob, worse_sent_prob, 
        better_rand_sent_prob, worse_rand_sent_prob, 
        better_worse, better_rand, rand_worse,
    ]:
        print('==============')
        for k, v in x.items():
            if not len(v): continue
            # print(k, sum(v)/max(1,len(v)))
            v.sort()
            cnt = len([x for x in v if x > 1])
            print(k, sum(v)/max(1, len(v)), v[len(v)//2], cnt/len(v), '|', len(v))
    
    print('==============')
    for k, v in clip_scores.items():
        if not len(v): continue
        # print(k, sum(v)/max(1,len(v)))
        v.sort()
        cnt = len([x for x in v if x > .6])
        print(k, sum(v)/max(1, len(v)), v[len(v)//2], cnt/len(v))
    
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
    
    # instructions = json_load(ptx_data_path)
    # inst_dict, qa_dict = cat_instructions(instructions, tokenizer)
    
    # preferences = json_load(data_path) + json_load(data_path2)
    # pref_dict = cat_preferences(preferences)
    
    # collect_data(inst_dict, pref_dict, qa_dict)
    
    # split_data('/home/users/nus/e0672129/scratch/LLaVA-DPO/data/all_preferences_33k_v10.json')
    
    # visualize_eval_rst(
    #     # '/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/eval/llava-v1.5-7b_eval_result.jsonl',
    #     '/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/eval-0rimg/llava-v1.5-7b_eval_result.jsonl',
    #     '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/all_preferences_4k_v10_foreval.json',
    #     tokenizer)
    
    # visualize_eval_rst('/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/eval-0rimg-rlhfv/llava-v1.5-7b_eval_result.jsonl',
    #                    '/home/users/nus/e0672129/scratch/RLHF-V/RLHF-V-Dataset-5.7k.json',
    #                    tokenizer)
    
    visualize_eval_rst('/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/eval-0rimg-llavarlhf/llava-v1.5-7b_eval_result.jsonl',
                       '/home/users/nus/e0672129/scratch/RLHF-V/llava-human-preference-10k.json',
                       tokenizer)
    
    # rawdata = json_load('/home/users/nus/e0672129/scratch/LLaVA-DPO/data/all_preferences_11k_v10.json')
    # data = []
    # for dt in tqdm(rawdata):
    #     if random.random() > 2/3: continue
    #     if 'images' in dt:
    #         if any(not os.path.exists(os.path.join('/home/users/nus/e0672129/scratch', x)) for x in dt['images']): continue
    #         if dt['images'][0] != dt['images'][1]:
    #             if dt['conversations'][-1]['value'].strip():
    #                 data.append(dt)
    #     elif 'image' in dt:
    #         if not os.path.exists(os.path.join('/home/users/nus/e0672129/scratch', dt['image'])): continue
    #         if dt['conversations'][-1]['value'] != dt['contrastive_conversations'][-1]['value']:
    #             if dt['conversations'][-1]['value'].strip() and dt['contrastive_conversations'][-1]['value'].strip():
    #                 data.append(dt)
    # json_dump(data, '/home/users/nus/e0672129/scratch/LLaVA-DPO/data/all_preferences_{}k_v10.json'.format(len(data) // 1000))
    