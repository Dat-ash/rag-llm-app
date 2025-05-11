import os
from langchain_openai import AzureChatOpenAI
import dotenv

dotenv.load_dotenv()

llm_stream = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview",
    azure_deployment = "gpt-4o",
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.3,
    streaming=True
    )

prompt = "Tell me something about Azure"

## Need to check whether invoke works or not?
# llm_stream.invoke(prompt)
for chunk in llm_stream.stream(prompt):
    print(chunk.content, end="", flush=True)