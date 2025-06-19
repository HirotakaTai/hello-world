from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = model | StrOutputParser()
    response = chain.invoke("Who are you?")
    print(response)


if __name__ == "__main__":
    main()
