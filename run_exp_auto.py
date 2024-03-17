import os
import subprocess
import time

# Set the threshold for the required free memory (in MB) to start the experiment
# MEMORY_THRESHOLD_MB = 40960
# MEMORY_THRESHOLD_MB = 46068
MEMORY_THRESHOLD_MB = 30720

def get_free_gpu_memory():
    """Get the free GPU memory in MB."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE)
    free_memory = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
    return free_memory


def run_experiment(gpu_ids):
    # Your experiment code here
    # completed_process = subprocess.run(["bash", '/home/yuxi/Projects/SGD/scripts/run_generation_gsm8k_llama.sh', 
    #                                     ','.join([str(x) for x in gpu_ids])],
    #                                    capture_output=True, text=True)
    
    # with open('stdout.log', 'w', encoding='utf-8') as f:
    #     f.write(completed_process.stdout)
    # with open('stderr.log', 'w', encoding='utf-8') as f:
    #     f.write(completed_process.stderr)
    completed_process = subprocess.run(['bash', '/home/yuxi/Projects/LLaVA-DPO/scripts/v1_5/eval/myeval.sh'])


if __name__ == "__main__":
    while True:
        free_memory = get_free_gpu_memory()
        sorted_memory = sorted(enumerate(free_memory), key=lambda x:x[1], reverse=True)
        sorted_memory = [x for x in sorted_memory if x[0] not in [1, 2, 3]]
        
        if sorted_memory[0][1] > MEMORY_THRESHOLD_MB - 1024:
            print("Running on gpu", sorted_memory[0][0])
            run_experiment([sorted_memory[0][0]])
            # break
        else:
            print("Waiting for enough GPU memory. Current free memory: {} GB".format(sum(free_memory) / 1024))
            print(sorted_memory)
            time.sleep(100)  # Check every 100 seconds

