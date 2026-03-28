import os
from getpass import getpass

from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = getpass("Enter GROQ API key: ").strip()
        os.environ["GROQ_API_KEY"] = api_key

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
    )

    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
    )

    print("Chatbot ready! Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = conversation.predict(input=user_input)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
