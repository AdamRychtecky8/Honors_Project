from PIL import Image
import pandas as pd
from openai import OpenAI
import base64
from tqdm import tqdm
from google import genai
import numpy as np
import time


def geminicomp(text,image_path,modname="gemini-2.0-flash-thinking-exp-01-21"):
    client = genai.Client(api_key = 'example")

    image = Image.open(image_path).convert('RGB')

    system_prompt = "You are being asked to judge who is more knowledgeable about an image given 2 descriptions. Your final response should only be \"Person A\" or \"Person B\", whichever is more knowledgeable. "
    response = client.models.generate_content(model=modname,contents=[system_prompt+text, image])

    text_out=response.text
    redo=False
    if len(text_out)>len('Answer: **Person B**'):
        time.sleep(6.1)
        print("REDOING")
        response = client.models.generate_content(model=modname,contents=[system_prompt+text, image])
        text_out=response.text
        redo=True

    return text_out,redo

llava=pd.read_excel('LLava_Outs.xlsx')
path="allimgs/"


df=np.load('gemini.npy',allow_pickle=True)[()]


for j in range(1,13):
    text=list(llava['Text'+str(j)])
    imgs=list(llava['Image'+str(j)])
    
    for i in tqdm(range(len(text))):
        if df[f'Output{j}'][i] == None:
            if str(text[i])=='0':
                df[f'Output{j}'][i] = '0'
            else:
                while True:
                    try:
                        start = time.time()
                        outputmodel,redo = geminicomp(text[i], path + str(imgs[i]))
                        end = time.time()
                        break  # Exit loop on success
                    except Exception as e:
                        if "500 INTERNAL" in str(e):
                            print("Server error encountered. Retrying in 5 seconds...")
                            time.sleep(12)
                        if "Connection aborted" in str(e):
                            print("Connection Aborted. Retrying in 300 seconds...")
                            time.sleep(300)
                        if "RESOURCE_EXHAUSTED" in str(e):
                            print("Exhausted.  Retrying in 300 seconds...")
                            time.sleep(300)
                        if "model is overload" in str(e):
                            print("Model Overloaded.  Retrying in 300 seconds...")
                            time.sleep(300)
                        else:
                            time.sleep(1000)
                df[f'Output{j}'][i] = str(outputmodel)
                end = time.time()
                timedone=end - start
                if timedone<6:
                    if redo:
                        time.sleep(12.2-timedone)
                    else:
                        time.sleep(6.1-timedone)


            np.save('gemini.npy',df)


and chatgpt:

def chatgptcomp(text,image_path,modname="deepseek-chat"):
    client = OpenAI(api_key = "example", base_url="https://api.deepseek.com")

    base64_image = encode_image(image_path)

    system_prompt = "You are R1, a large language model trained by DeepSeek. You are being asked to judge who is more knowledgeable about an image given 2 descriptions. Only respond with Person A or Person B, whichever is more knowledgeable."

    response = client.chat.completions.create(
        model=modname,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": text,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                ],
            },
            
        ]    )

    # Extract the assistant's reply from the response
    return response.choices[0].message.content