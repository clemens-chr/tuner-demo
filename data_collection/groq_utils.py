import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def groq(prompt, image=None):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a chatbot that is used to instruct the user on how to record a short datasaet (fo around 10 videos) to finetune a visual to policy model used in robotics. Use bulletpoints and use emojis so its nice. Don't make it too long."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content