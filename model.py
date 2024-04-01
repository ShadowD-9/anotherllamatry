import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

You task is to suggest recipes and give people information about nutritional information about their food based on the Dietary Quality Index (DQI)
The necessary nutritional values to take into account for the DQI and for the recipe suggestions are: Protein (g), Lipids (g), Fiber (g), Ascorbic Acid (mg), Cholesterol (mg), Saturated Fatty Acids (g), Calcium (mg), Iron (mg), Sodium (mg), Carbohydrates (g), SFAs, MFAs, PUFAs and Total Energy (Kcal)

To score the food based on the DQI there are 4 groups (Variety, Adequacy, Moderation and Overall balance), the scores are for daily intake. Variety takes a maximum score of 20, Adequacy a maximum score of 40, Moderation a maximum score of 30 and Overall Balance a maximum score of 10 so the DQI has a maxim score of 100

For Variety there are 2 subgroups, Overall food group variety (meat/poultry/fish/eggs; dairy/beans; grain; fruit; vegetable) that gives a maximum score of 15 if 1 food from each group was consumed, for each group that was not consumed 3 points are taken from the total. The other subgroup is Within-group variety for protein source (meat/poultry/fish; dairy, beans, eggs), if food was consumed from 3 or more different sources it gives a score of 5, for each less source, it takes away 2 points

For Adequacy there are 8 subgroups, for vegetables group if there was 3 to 5 servings in a day, it gives 5 points, for fruit groups if there was 2 to 4 servings it gives 5 points, for grain groups 6 to 11 servings a day give 5 points, for fiber 20 to 30 grams a day give 5 points, for protein if it is 10% or more of the energy (Kcal) it took in a day it give 5 points, for iron if it is 100% of the necessary RDA (AI) for a person in a day it give 5 points, for Calcium if it is  100% AI in a day it give 5 points and for vitamin C if it was 100% RDA (RNI) in a day it give 5 points, for each group if there was less than expected it give a proportional value being a whole number 

For Moderation there are 5 subgroups, Total fat give 6 points if it was 20% or less of total energy in a day, if it was between more than 20% to 30% it gives 3 points and else it gives 0 points, for Saturated fat if it was 7% or less of total energy in a day it gives 6 points, between more than 7% to 10% it give 3 points, else 0 points, for Cholesterol if 300 mg or less were consumed it gives 6 points, from more than 300 to 400 mg it give 3 points, else 0 points, for Sodium if 2400 mg or less were consumed it give 6 points, between more than 2400 and 3400 it give 3 points, else 0 points and finally for Empty calorie food if it is 3% of the total energy or less it give 6 points, between more than 3% to10% it gives 3 points, else 0 points

For Overall balance there are 2 subgroups, Macronutrient ratio (carbohydrate:protein:fat), if carbohydrate was consumed between 55 to 65% of the total energy (Kcal) in a day and protein was consumed between 10 to 15% total energy in a day and fat was consumed between 15 to 25% total energy in a day it give 6 points, if the consumption for carbohydrates was between 52 to 68%, protein between 9 to 16 and fat between13 to 27 it give 4 points, if the consumption for carbohydrates was between 50 to 70% and protein between 8 to 17% and fat between 12 to 30% it give 2 points, else 0 points, the other subgroup is Fatty acid ratio (PUFA(P):MUFA(M):SFA(S)), if P/S is between1 to 1.5 and M/S is between1 to 1.5 it give 4 points, else if P/S is between 0.8 to 1.7 and M/S is between 0.8 to 1.7 it give 2 point, else 0 points.


<</SYS>>

Suggest me a nutritional recipe for today as breakfast, lunch and dinner that gives me the nutritional intake as said in DQI [/INST]
"""
DATA_PATH="data/"
DB_FAISS_PATH="vectorstores/db_faiss"

def create_faiss_vector_db(data_path, db_faiss_path):
    """
    Create FAISS vector database from PDF documents.

    Parameters:
    - data_path (str): Path to the directory containing PDF documents.
    - db_faiss_path (str): Path to save the FAISS vector database.
    """
    loader = CSVLoader(DATA_PATH, encoding="utf-8", csv_args={
                'delimiter': ','})
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Enable dangerous deserialization
    db = FAISS.load_local(db_faiss_path, embeddings, allow_dangerous_deserialization=True)
    if db is None:
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(db_faiss_path)


def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    # Implement the question-answering logic here
    response = qa({'query': query})
    return response['result']

def add_vertical_space(spaces=1):
    for _ in range(spaces):
        st.markdown("---") 

def main():
    st.set_page_config(page_title="Llama-2 Recipe Chatbot")

    with st.sidebar:
        st.title('Llama-2 Recipe Chatbot')
        st.markdown('''
        ## About
                    
        This Chatbot powered by Llama-2-7B suggest different recipes depending on the likes of the user, it will give nutritional recipes for you daily intake based on the Dietary Quality Index (DQI).

        Tips for using:
        Can you give me a recipe and its nutritional value?

        Give me recipes to make this day, from breakfast, lunch and dinner that will give me the necessary nutritional intake for the whole day

        I have eggs, red peppers and ham slices, what recipe could you give me with those ingredients that gives me a good nutritional intake for my breakfast

        I am vegan so could you give me a recipe without any meat nor milk or any animal based product that is nutritious?


        ''')
        add_vertical_space(1)  # Adjust the number of spaces as needed
        st.write('This is test 1 of the chatbot, it is still a work in progress')
        add_vertical_space(1)  # Adjust the number of spaces as needed
        st.write('Made by --------')

    st.title("Llama-2 Recipe Chatbot")
    st.markdown(
        """
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                height: 400px;
                overflow-y: auto;
                padding: 10px;
                color: white; /* Font color */
            }
            .user-bubble {
                background-color: #007bff; /* Blue color for user */
                align-self: flex-end;
                border-radius: 10px;
                padding: 8px;
                margin: 5px;
                max-width: 70%;
                word-wrap: break-word;
            }
            .bot-bubble {
                background-color: #363636; /* Slightly lighter background color */
                align-self: flex-start;
                border-radius: 10px;
                padding: 8px;
                margin: 5px;
                max-width: 70%;
                word-wrap: break-word;
            }
        </style>
        """
    , unsafe_allow_html=True)

    conversation = st.session_state.get("conversation", [])
    
    query = st.text_input("Ask your question here:", key="user_input")
    if st.button("Get Answer"):
        if query:
            with st.spinner("Processing your question..."):  # Display the processing message
                conversation.append({"role": "user", "message": query})
                # Call your QA function
                answer = qa_bot(query)
                conversation.append({"role": "bot", "message": answer})
                st.session_state.conversation = conversation
        else:
            st.warning("Please input a question.")

    chat_container = st.empty()
    chat_bubbles = ''.join([f'<div class="{c["role"]}-bubble">{c["message"]}</div>' for c in conversation])
    chat_container.markdown(f'<div class="chat-container">{chat_bubbles}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
