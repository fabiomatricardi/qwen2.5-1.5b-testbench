# Chat with an intelligent assistant in your terminal  
# MODEL: https://huggingface.co/DavidAU/Gemma-The-Writer-9B-GGUF
# model/Gemma-The-Writer-9B-D_AU-Q4_k_m.gguf
import sys
from time import sleep
import warnings
warnings.filterwarnings(action='ignore')
import datetime
import random
import string
import tiktoken
import argparse

#Add GPU argument in the parser
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", action="store_true")

args = parser.parse_args()
GPU = args.gpu
if GPU:
    ngpu_layers = 28
    print(f'Selected GPU: offloading {ngpu_layers} layers...')
else:
     ngpu_layers = 0   
     print('Loading Model on CPU only......')

encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))
modelname = 'qwen2.5-1.5b-instruct'
def countTokens(text):
    encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))
    numoftokens = len(encoding.encode(text))
    return numoftokens

def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res

# create THE LOG FILE 
logfile = f'{genRANstring(5)}_log.txt'
logfilename = logfile
#Write in the history the first 2 sessions
writehistory(logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT with 💻 {modelname}\n---\n🧠🫡: You are a helpful assistant.')    
writehistory(logfilename,f'💻: How can I assist you today in writing?')

print("\033[95;3;6m")
print("1. Waiting 10 seconds for the API to load...")
from llama_cpp import Llama
llm = Llama(
            model_path='qwen2.5-1.5b-instruct-q5_k_m.gguf',
            n_gpu_layers=ngpu_layers,
            temperature=1.1,
            top_p = 0.5,
            n_ctx=8192,
            max_tokens=1500,
            repeat_penalty=1.178,
            stop=['<|im_end|>'],
            verbose=False,
            )
print("2. Model qwen2.5-1.5b-instruct-q5_k_m.gguf loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = []
print("\033[92;1m")
print(f'📝Logfile: {logfilename}')
counter = 1
while True:
    # Reset history after 3 turns
    if counter > 5:
        history = []        
    userinput = ""
    print("\033[1;30m")  #dark grey
    print("Enter your text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[91;1m")  #red
    lines = sys.stdin.readlines()
    for line in lines:
        userinput += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break
    history.append({"role": "user", "content": userinput})
    print("\033[92;1m")
    # Preparing Generation history pair
    new_message = {"role": "assistant", "content": ""}
    # Starting generation loop
    full_response = ""
    fisrtround = 0
    start = datetime.datetime.now()
    print("💻 > ", end="", flush=True)
    for chunk in llm.create_chat_completion(
        messages=history,
        temperature=1.2,
        repeat_penalty= 1.178,
        stop=['<|im_end|>'],
        max_tokens=1500,
        stream=True,):
        try:
            if chunk["choices"][0]["delta"]["content"]:
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                full_response += chunk["choices"][0]["delta"]["content"]                              
        except:
            pass        
    new_message["content"] = full_response
    history.append(new_message)  
    counter += 1  
    delta = datetime.datetime.now() - start
    totalseconds = delta.total_seconds()
    output = full_response
    prompttokens = countTokens(userinput)
    assistanttokens = countTokens(output)
    totaltokens = prompttokens + assistanttokens
    speed = totaltokens/totalseconds
    genspeed = assistanttokens/totalseconds
    print('')
    print("\033[91;1m")
    rating = input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
    print("\033[92;1m")
    stats = f'''---
Prompt Tokens: {prompttokens}
Output Tokens: {assistanttokens}
TOTAL Tokens: {totaltokens}
>>>⏱️Inference time: {delta}
>>>🧮Inference speed: {speed:.3f}  t/s
>>>🏃‍♂️Generation speed: {genspeed:.3f}  t/s
📝Logfile: {logfilename}
>>>💚User rating: {rating}

'''
    print(stats)
    writehistory(logfilename,f'''👨‍💻: {userinput}
💻 > {full_response}
{stats}
📝Logfile: {logfilename}''')
