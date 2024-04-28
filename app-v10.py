import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import openai
import os
import nest_asyncio
from llama_parse import LlamaParse
from dotenv import load_dotenv
import os



def get_pdf_text(pdf_docs):
    parsingInstructionTable = """Dit is een studiehandleiding met verschillende tabellen.
    Bijvoorbeeld worden competenties met bijbehorende niveaus opgesplitst in afzonderlijke tabellen. Hier is de eerste regel een merged cel is met de beschrijving van de competentie - bijvoorbeeld
    9. Handelen vanuit waarden: De startende bedrijfskundige professional handelt vanuit een waardenbesef en heeft bij het zoeken naar oplossingen voor bedrijfskundige vraagstukken oog voor mogelijke consequenties van (bewuste of onbewuste) keuzes en handelingen op de langere termijn.
    Daaronder volgen drie cellen met de niveautitels: Niveau 1 | Niveau 2 | Niveau 3
    Daaronder de beschrijvingen van ieder niveau: Kan handelen volgens algemeen aanvaarde of professionele sociale en ethische normen en waarden. Gaat daarbij zorgvuldig en discreet om met informatie over personen en/of de organisatie. | Heeft zicht op de mogelijke consequenties van bedrijfskundig handelen voor stakeholders, milieu en maatschappij. | Legt verantwoording af over gemaakte keuzes op basis van het eigen morele kompas als professioneel normerende kaders. Streeft aantoonbaar naar duurzame maatschappelijke verantwoordde bedrijfskundige oplossingen."""

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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# def get_vectorstore(text_chunks):
#     embeddings = CustomOpenAIEmbeddings()
#     vectorstore = FAISS()

#     # Generate embeddings for each chunk
#     for chunk in text_chunks:
#         chunk_embedding = embeddings.embed([chunk])[0]  # Assuming embed function returns a list of embeddings
#         vectorstore.add([chunk_embedding])  # Add the embedding to the vector store

#     return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
