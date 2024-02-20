import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from faithscore.framework import FaithScore

images = ["/mnt/data/yuxi/coco/coco2014/val2014/COCO_val2014_000000164255.jpg"]
answers = ["The main object in the image is a colorful beach umbrella."]

scorer = FaithScore(
                    vem_type="ofa", 
                    api_key="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", 
                    llava_path="liuhaotian/llava-v1.5-7b", 
                    use_llama=False,)
score, sentence_score = scorer.faithscore(answers, images)