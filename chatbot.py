import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")
chat = model.start_chat(history=[])

print("AI Chatbot type quit to exit")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Exiting the chatbot. Goodbye!")
        break

    response = chat.send_message(user_input)
    print("AI: ", response.text)
    print('-' * 50)