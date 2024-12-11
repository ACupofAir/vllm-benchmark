import torch
import time
import argparse
from transformers import GPT2ForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Qwen2ForSequenceClassification
from torch.profiler import profile, record_function, ProfilerActivity, schedule


# Get the batch size and device
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', type=int, default=1, help='an integer for the batch size')
parser.add_argument('--device', type=str, default='cpu', help='an string for the device')
parser.add_argument('--profile', type=bool, default=False, help='enable protch profiler for CPU/XPU')
parser.add_argument('--engine', type=str, default='ipex-llm', help='an string for the device')
parser.add_argument('--prompt', type=int, default=1024, help='an int for prompt length')
args = parser.parse_args()
enable_profile=args.profile
batch_size = args.batch_size
device = args.device
engine = args.engine
prompt_len = args.prompt
print(f"The batch size is: {batch_size}, device is {device}, prompt is {prompt_len}")

################################################################################
ENGLISH_PROMPT="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. However, her parents were always telling her to stay close to home, to be careful, and to avoid any danger. But the little girl was stubborn, and she wanted to see what was on the other side of the mountain. So she sneaked out of the house one night, leaving a note for her parents, and set off on her journey. As she climbed the mountain, the little girl felt a sense of excitement and wonder. She had never been this far away from home before, and she couldnt wait to see what she would find on the other side. She climbed higher and higher, her lungs burning from the thin air, until she finally reached the top of the mountain. And there, she found a beautiful meadow filled with wildflowers and a sparkling stream. The little girl danced and played in the meadow, feeling free and alive. She knew she had to return home eventually, but for now, she was content to enjoy her adventure. As the sun began to set, the little girl reluctantly made her way back down the mountain, but she knew that she would never forget her adventure and the joy of discovering something new and exciting. And whenever she felt scared or unsure, she would remember the thrill of climbing the mountain and the beauty of the meadow on the other side, and she would know that she could face any challenge that came her way, with courage and determination. She carried the memories of her journey in her heart, a constant reminder of the strength she possessed. The little girl returned home to her worried parents, who had discovered her note and anxiously awaited her arrival. They scolded her for disobeying their instructions and venturing into the unknown. But as they looked into her sparkling eyes and saw the glow on her face, their anger softened. They realized that their little girl had grown, that she had experienced something extraordinary. The little girl shared her tales of the mountain and the meadow with her parents, painting vivid pictures with her words. She spoke of the breathtaking view from the mountaintop, where the world seemed to stretch endlessly before her. She described the delicate petals of the wildflowers, vibrant hues that danced in the gentle breeze. And she recounted the soothing melody of the sparkling stream, its waters reflecting the golden rays of the setting sun. Her parents listened intently, captivated by her story. They realized that their daughter had discovered a part of herself on that journey—a spirit of curiosity and a thirst for exploration. They saw that she had learned valuable lessons about independence, resilience, and the beauty that lies beyond ones comfort zone. From that day forward, the little girls parents encouraged her to pursue her dreams and embrace new experiences. They understood that while there were risks in the world, there were also rewards waiting to be discovered. They supported her as she continued to embark on adventures, always reminding her to stay safe but never stifling her spirit. As the years passed, the little girl grew into a remarkable woman, fearlessly exploring the world and making a difference wherever she went. The lessons she had learned on that fateful journey stayed with her, guiding her through challenges and inspiring her to live life to the fullest. And so, the once timid little girl became a symbol of courage and resilience, a reminder to all who knew her that the greatest joys in life often lie just beyond the mountains we fear to climb. Her story spread far and wide, inspiring others to embrace their own journeys and discover the wonders that awaited them. In the end, the little girls adventure became a timeless tale, passed down through generations, reminding us all that sometimes, the greatest rewards come to those who dare to step into the unknown and follow their hearts. With each passing day, the little girls story continued to inspire countless individuals, igniting a spark within their souls and encouraging them to embark on their own extraordinary adventures. The tale of her bravery and determination resonated deeply with people from all walks of life, reminding them of the limitless possibilities that awaited them beyond the boundaries of their comfort zones. People marveled at the little girls unwavering spirit and her unwavering belief in the power of dreams. They saw themselves reflected in her journey, finding solace in the knowledge that they too could overcome their fears and pursue their passions. The little girl's story became a beacon of hope, a testament to the human spirit"

def get_fixed_length_string(input_string, token_length, tokenizer):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    input_ids = input_ids[:, :token_length]
    return tokenizer.batch_decode(input_ids)[0]

################################################################################

######################################################################################
# PyTorch Profiling with IPEX
# export IPEX_ZE_TRACING=1
# export ZE_ENABLE_TRACING_LAYER=1
import contextlib
def profiler_setup(profiling=False, *args, **kwargs):
    if profiling:
        return torch.profiler.profile(*args, **kwargs)
    else:
        return contextlib.nullcontext()

my_schedule = schedule(
    skip_first=6,
    wait=1,
    warmup=1,
    active=1
    )

# also define a handler for outputing results
def trace_handler(p):
    if(device == 'xpu'):
        print(p.key_averages().table(sort_by="self_xpu_time_total", row_limit=20))
    print(p.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # p.export_chrome_trace("./trace_" + str(p.step_num) + ".json")
#######################################################################################


dtype = torch.bfloat16 if device == 'cpu' else torch.float16
num_labels = 5

model_name="/llm/models/gpt2-medium-classification"
# model_name="/home/llm/local_models/Qwen/Qwen2-0.5B-Instruct"

#model_name = model_name + "-classification"
model_name_ov = model_name + "-ov"
# model_name_ov = model_name_ov + "-int8"
model_name_ov = model_name_ov + "-fp16"

if (engine == 'ipex') :
    import torch
    import intel_extension_for_pytorch as ipex
    # Need padding from the left and padding to 1024
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype,
                                                               pad_token_id=tokenizer.eos_token_id,
                                                               low_cpu_mem_usage=True
                                                               ).eval().to(device)
elif (engine == 'ipex-llm'):
    from ipex_llm.transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               torch_dtype=dtype,
                                                               load_in_low_bit="fp16",
                                                               pad_token_id=tokenizer.eos_token_id,
                                                               low_cpu_mem_usage=True).to(device)
    model = torch.compile(model, backend='inductor')
    print(model)
else:
    from optimum.intel import OVModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name_ov, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = OVModelForSequenceClassification.from_pretrained(model_name_ov, torch_dtype=dtype).to(device)



# Intel(R) Extension for PyTorch*
if engine == 'ipex':
    if device == 'cpu':
        # model = ipex.llm.optimize(model, dtype=dtype, inplace=True, deployment_mode=True)
        # ############## TorchDynamo ###############
        model = ipex.optimize(model, dtype=torch.bfloat16, weights_prepack=False)
        model = torch.compile(model, backend='ipex')
        # ##########################################
    else:    # Intel XPU
        #model = ipex.llm.optimize(model, dtype=dtype, device="xpu", inplace=True)
        model = ipex.optimize(model, dtype=dtype, inplace=True)

    model=torch.compile(model, backend="inductor")
    print(model)

    # # #######calulate the total num of parameters########
    # def model_size(model):
    #     return sum(t.numel() for t in model.parameters())
    # print(f"GPT2 size: {model_size(model)/1000**2:.1f}M parameters")
    # # # #######print model information  ###################
    # print(model)

    # ########Enable the BetterTransformer  ###################
    # only Better Transformer only support GPT2, not support Qwen2
    # model = BetterTransformer.transform(model)
#elif engine == 'ipex-llm':
#    model = ipex.optimize(model, dtype=dtype, inplace=True)
#    model=torch.compile(model) #backend="inductor")
elif engine == 'ov':
    print("OV inference")


#prompt = ["this is the first prompt"]

prompts = get_fixed_length_string(ENGLISH_PROMPT, prompt_len, tokenizer)
print(prompts)

# Tokenize the batch of prompts
inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
# print(inputs)

if engine == 'ipex' or engine == 'ipex-llm':
    #ipex need move the inputs to device, but OV doesn't need
    inputs.to(device)

    # Initialize an empty list to store elapsed times
    elapsed_times = []

    # Loop for batch processing 10 times and calculate the time for every loop
    with profiler_setup(profiling=enable_profile, activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        schedule=my_schedule,
        on_trace_ready=trace_handler,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/gpt2'),
        record_shapes=True,
        with_stack=True
        ) as prof:

        for i in range(10):
            start_time = time.time()

            # Perform inference
            with torch.inference_mode():
                # logits = model(**inputs).logits
                outputs = model(**inputs)
                logits = outputs.logits

            # Get the predicted class for each input in the batch
            predicted_class_ids = logits.argmax(dim=1).tolist()

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Save the elapsed time in the list
            elapsed_times.append(elapsed_time)

            if(enable_profile):
                prof.step()

            # print(outputs)
            # print(type(outputs))
            # print("logits.shape is " + str(logits.shape))
            # print(logits)

            # print(predicted_class_ids)

elif engine == 'ov':
    print("OV inference")
        # Initialize an empty list to store elapsed times
    elapsed_times = []

    # Loop for batch processing 10 times and calculate the time for every loop
    for i in range(10):
        start_time = time.time()

        outputs = model(**inputs)
        logits = outputs.logits

        # Get the predicted class for each input in the batch
        predicted_class_ids = logits.argmax(dim=1).tolist()

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Save the elapsed time in the list
        elapsed_times.append(elapsed_time)

        # print(outputs)
        # print(type(outputs))
        # print("logits.shape is " + str(logits.shape))
        # print(logits)

        # print(predictions)
        #print(predicted_class_ids)


# Skip the first two values and calculate the average of the remaining elapsed times
average_elapsed_time = sum(elapsed_times[2:]) / len(elapsed_times[2:])
classfication_per_second = batch_size/average_elapsed_time
print(f"Average time taken (excluding the first two loops): {average_elapsed_time:.4f} seconds, Classification per seconds is {classfication_per_second:.4f}")
