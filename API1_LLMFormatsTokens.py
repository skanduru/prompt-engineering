#!/usr/bin/env python
# coding: utf-8

# # L1 Language Models, the Chat Format and Tokens

# ## Setup
# #### Load the API key and relevant Python libaries.
# In this course, we've provided some code that loads the OpenAI API key for you.

# In[ ]:


import os
import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


# #### helper function
# This may look familiar if you took the earlier course "ChatGPT Prompt Engineering for Developers" Course

# In[ ]:


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


# ## Prompt the model and get a completion

# In[ ]:


response = get_completion("What is the capital of France?")


# In[ ]:


print(response)


# ## Tokens

# In[ ]:


response = get_completion("Take the letters in lollipop \
and reverse them")
print(response)


# "lollipop" in reverse should be "popillol"

# In[ ]:


response = get_completion("""Take the letters in \
l-o-l-l-i-p-o-p and reverse them""")


# In[ ]:


response


# ## Helper function (chat format)
# Here's the helper function we'll use in this course.

# In[ ]:


def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, # the maximum number of tokens the model can ouptut 
    )
    return response.choices[0].message["content"]


# In[ ]:


messages =  [  
{'role':'system', 
 'content':"""You are an assistant who\
 responds in the style of Dr Seuss."""},    
{'role':'user', 
 'content':"""write me a very short poem\
 about a happy carrot"""},  
] 
response = get_completion_from_messages(messages, temperature=1)
print(response)


# In[ ]:


# length
messages =  [  
{'role':'system',
 'content':'All your responses must be \
one sentence long.'},    
{'role':'user',
 'content':'write me a story about a happy carrot'},  
] 
response = get_completion_from_messages(messages, temperature =1)
print(response)


# In[ ]:


# combined
messages =  [  
{'role':'system',
 'content':"""You are an assistant who \
responds in the style of Dr Seuss. \
All your responses must be one sentence long."""},    
{'role':'user',
 'content':"""write me a story about a happy carrot"""},
] 
response = get_completion_from_messages(messages, 
                                        temperature =1)
print(response)


# In[ ]:


def get_completion_and_token_count(messages, 
                                   model="gpt-3.5-turbo", 
                                   temperature=0, 
                                   max_tokens=500):
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    
    content = response.choices[0].message["content"]
    
    token_dict = {
'prompt_tokens':response['usage']['prompt_tokens'],
'completion_tokens':response['usage']['completion_tokens'],
'total_tokens':response['usage']['total_tokens'],
    }

    return content, token_dict


# In[ ]:


messages = [
{'role':'system', 
 'content':"""You are an assistant who responds\
 in the style of Dr Seuss."""},    
{'role':'user',
 'content':"""write me a very short poem \ 
 about a happy carrot"""},  
] 
response, token_dict = get_completion_and_token_count(messages)


# In[ ]:


print(response)


# In[ ]:


print(token_dict)


# #### Notes on using the OpenAI API outside of this classroom
# 
# To install the OpenAI Python library:
# ```
# !pip install openai
# ```
# 
# The library needs to be configured with your account's secret key, which is available on the [website](https://platform.openai.com/account/api-keys). 
# 
# You can either set it as the `OPENAI_API_KEY` environment variable before using the library:
#  ```
#  !export OPENAI_API_KEY='sk-...'
#  ```
# 
# Or, set `openai.api_key` to its value:
# 
# ```
# import openai
# openai.api_key = "sk-..."
# ```

# #### A note about the backslash
# - In the course, we are using a backslash `\` to make the text fit on the screen without inserting newline '\n' characters.
# - GPT-3 isn't really affected whether you insert newline characters or not.  But when working with LLMs in general, you may consider whether newline characters in your prompt may affect the model's performance.
