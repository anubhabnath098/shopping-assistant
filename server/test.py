import time, os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
m = genai.GenerativeModel(os.environ["GEMINI_MODEL_NAME"])
t = time.perf_counter()
try:
    print(m.generate_content("Say hi", request_options={"retry": None, "timeout": 100}).text)
except Exception as e:
    print(type(e).__name__, e)
print(f"{time.perf_counter() - t:.2f}s")