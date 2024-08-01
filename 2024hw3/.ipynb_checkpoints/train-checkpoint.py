# Import packages
import google.generativeai as genai
from typing import List, Tuple
import gradio as gr
import json

# Set up Gemini API key
## TODO: Fill in your Gemini API in the ""
GOOGLE_API_KEY="AIzaSyCNbSjSFIT68Wf5va4ErmKvlPBe9ysrK9M"
genai.configure(api_key=GOOGLE_API_KEY, transport='rest')

model = genai.GenerativeModel('gemini-pro')

# Check if you have set your Gemini API successfully
# You should see "Set Gemini API sucessfully!!" if nothing goes wrong.
try:
    model.generate_content(
      "test",
    )
    print("Set Gemini API sucessfully!!")
except:
    print("There seems to be something wrong with your Gemini API. Please follow our demonstration in the slide to get a correct one.")