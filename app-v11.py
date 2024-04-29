import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from llama_parse import LlamaParse

load_dotenv()


def get_pdf_text():
    parsingInstructionTable = """Dit is een studiehandleiding met verschillende tabellen.Hoe de informatie in tabellen zoveel mogelijk bij elkaar."""

    parser = LlamaParse(
        # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="nl",  # Optionally you can define a language, default=en
        parsing_instruction=parsingInstructionTable
    )

    # async
    documents = parser.load_data("./doc.pdf")
    text = documents[0].text
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()

    # model_name = "hkunlp/instructor-xl"
    # model_kwargs = {'device': 'cpu'}
    # encode_kwargs = {'normalize_embeddings': False}
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )
    embeddings = embeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_response(user_query, chat_history, vectorstore):
    
    llm = ChatOpenAI()
    llm = llm
    retriever = vectorstore.as_retriever()

    ### Contextualize question ###
    contextualize_q_system_prompt = """Gezien de chat history en de meest recente user query, die mogelijk verwijst \
    naar de context in de chat history, formuleer een op zichzelf staande vraag die begrijpelijk is \
    zonder de chat history. Beantwoord de vraag NIET, herschrijf deze indien nodig en geef  \
    deze anders ongewijzigd terug."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever_chain = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """Je bent een assistent voor het beantwoorden van vragen. \
        Gebruik de volgende opgehaalde contextstukken om de vraag te beantwoorden. \
        Als je het antwoord niet weet, zeg dan gewoon dat je het niet weet.\

        {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    
    result = retrieval_chain.invoke(
        {"input": user_query, "chat_history": chat_history},
    )["answer"]

    return result

def main():

    # get pdf text
    raw_text = get_pdf_text()

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # app config
    st.set_page_config(page_title="Streaming bot", page_icon="🤖")
    st.title("Streaming bot")

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hallo! Ik ben een streaming bot. Waarmee kan ik je helpen?"),
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input("Stel je vraag hier: .....")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = get_response(user_query, st.session_state.chat_history, vectorstore)
            st.write(response)

        st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()        