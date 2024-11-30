import os
from io import BytesIO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import Document

# Directories for vector store and uploaded files
VECTOR_STORE_DIR = "vector_store"
UPLOADED_FILES_DIR = "uploaded_files"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(UPLOADED_FILES_DIR, exist_ok=True)

# Initialize LangChain components
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
combine_docs_chain = load_qa_chain(llm=llm, chain_type="refine")

# Initialize Streamlit
st.set_page_config(page_title="CHAT with your DATA", layout="wide")
st.title("🧠 CHAT with your DATA")

# Helper Function: List files with metadata
# Function to list files with metadata
def list_files_with_metadata(directory):
    """
    List files in the directory along with metadata like file type and size.
    Returns a Pandas DataFrame.
    """
    try:
        files_data = []
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                file_type = os.path.splitext(file_name)[-1].lower()  # Get file extension
                file_size = os.path.getsize(file_path)  # Get file size in bytes
                files_data.append({"File Name": file_name, "File Type": file_type, "Size (KB)": round(file_size / 1024, 2)})
        return pd.DataFrame(files_data)
    except FileNotFoundError:
        st.sidebar.error(f"Directory '{directory}' not found.")
        return pd.DataFrame()

# Generate a DataFrame for files in the uploaded files directory
files_df = list_files_with_metadata(UPLOADED_FILES_DIR)

# Display the DataFrame in the sidebar sorted by index and type of file
st.sidebar.subheader("📄 Datastore Files")
if not files_df.empty:
    # Reset and sort DataFrame by index and File Type
    files_df = files_df.sort_values(by=["File Type"]).reset_index(drop=True)  # Sort by File Type
    files_df.index += 1  # Set index to start from 1 for display purposes
    files_df = files_df.sort_index()  # Sort by the DataFrame's index
    
    # Display the table with a scrollbar if rows exceed six
    st.sidebar.dataframe(files_df, height=200)  # Set height to display max 6 rows with scrollbar
else:
    st.sidebar.write("No files found in the directory.")
#---------------------------------------------------------------------------   

# Initialize Vector Store
vector_store = Chroma(embedding_function=embeddings, persist_directory=VECTOR_STORE_DIR)

# Helper Function: Check if Vector Store is Empty
def is_vector_store_empty(vector_store):
    try:
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 2})
        documents = retriever.get_relevant_documents("")
        return len(documents) == 0
    except Exception:
        return True

vector_store_empty = is_vector_store_empty(vector_store)

# Helper Function: Get Unique Document Names
def get_unique_document_names(vector_store):
    try:
        collection = vector_store._collection
        results = collection.get(include=['metadatas'])
        document_names = set()
        for metadata in results['metadatas']:
            if 'document_name' in metadata:
                document_names.add(metadata['document_name'])
        return list(document_names)
    except Exception as e:
        st.error(f"Error retrieving document names: {e}")
        return []


# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_chat_history" not in st.session_state:
    st.session_state.show_chat_history = True  # Default to showing chat history

# Initialize or Load Vector Store
if os.path.exists(VECTOR_STORE_DIR):
    st.sidebar.info("Loading existing vector store...")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    vector_store_empty = False
else:
    st.sidebar.info("Creating a new vector store...")
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    vector_store_empty = True

# Check if vector store is empty
def is_vector_store_empty(vector_store):
    try:
        retriever = vector_store.as_retriever(search_type="mmr",
                                   search_kwargs={'k': 2, 'fetch_k': 50}
                                    )
         
        documents = retriever.get_relevant_documents("")
        return len(documents) == 0
    except Exception:
        return True

if not vector_store_empty:
    vector_store_empty = is_vector_store_empty(vector_store)


# Sidebar: Dynamic content based on active tab
def update_sidebar(active_tab):
    
    if active_tab == "📄 PDF Agent":
        # Sidebar: Chat History Options
        st.sidebar.subheader("💬 Chat Options")
        st.session_state.show_chat_history = st.sidebar.checkbox(
            "Show Chat History", value=st.session_state.show_chat_history
        )
        if st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.sidebar.success("Chat history cleared!")
        
    elif active_tab == "📊 CSV Agent":
        st.sidebar.subheader("CSV Agent Options")
        uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOADED_FILES_DIR, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved {uploaded_file.name} to uploaded files.")

            
            
    elif active_tab == "⚙️ General Agent":
        st.sidebar.subheader("General Options")
        st.sidebar.write("Perform general tasks.")
        st.sidebar.text_input("General Input:")
    else:
        st.sidebar.write("No options available for this tab.")


# Tabs for different agents
tabs = tabs = ["📄 PDF Agent", "📊 CSV Agent", "⚙️ General Agent"]
active_tab = st.radio("Select a Tab", tabs, key="active_tab", horizontal=True)

# Dynamically update the sidebar based on the active tab
update_sidebar(active_tab)

# --------------------------- PDF Agent Tab ---------------------------
if active_tab == "📄 PDF Agent":
    st.subheader("Ask Questions")
    st.markdown(
    """
    Enter your question below to query the uploaded files. 
    Results will be retrieved based on the content stored in the vector store.
    """
     )
    query = st.text_area("🧐 Ask a question:")

    # **Allow the user to select documents to query**
    document_names = get_unique_document_names(vector_store)
    if document_names:
        selected_documents = st.multiselect(
            "Select documents to query (leave empty to query all):",
            options=document_names,
            default=document_names
        )
    else:
        selected_documents = []

    use_contextual_search = st.checkbox("Use contextual retrieval (prioritize similar files)?", value=True)

    if query:
        if vector_store_empty:
            st.error("The vector store is empty. Please upload PDF files first.")
        else:
            with st.spinner("Retrieving the answer..."):
                if selected_documents and set(selected_documents) != set(document_names):
                    retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={'k': 2, 'filter': {'document_name': {'$in': selected_documents}}}
                    )
                else:
                    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 2})
                
                # You are geologist.
                #template = """Answer the question based on the following context:
                #              {context}

                #              Question: {question}
                #              """
                #question_prompt = PromptTemplate.from_template(template)
                
                        # Add conversational memory (optional)
                question_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="Given the following context: {context}, refine the question: {question}"
                )
                
                memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                                input_key="question",
                                                output_key='answer',
                                                k=3,
                                                return_messages=True   # Explicitly specify which key to store in memory
                                                )

                question_generator_chain = LLMChain(
                    llm=llm,
                    prompt=question_prompt,
                    memory=memory
                )
                    
                    # Create the ConversationalRetrievalChain
                qa_chain = ConversationalRetrievalChain(
                    retriever=retriever,
                    combine_docs_chain=combine_docs_chain,
                    question_generator=question_generator_chain,
                    memory=memory,  # Add memory here
                    return_source_documents=True,
                    return_generated_question=True
                )
                result = qa_chain({"question": query})
                answer = result['answer']
                source_docs = result['source_documents']

                # Gather metadata: document name and pages
                metadata_info = []
                for doc in source_docs:
                    document_name = doc.metadata.get('document_name', 'Unknown Document')  # Default if no name exists
                    page_number = doc.metadata.get('page', 'Unknown Page')  # Default if no page number exists
                    metadata_info.append(f"Document: {document_name}, Page: {page_number}")

                # Combine metadata into a single string
                metadata_summary = "\n".join(metadata_info)

                # Display the result
                st.write("### 🤖 Answer:")
                st.write(answer)

                if metadata_info:
                    st.write("### 📄 Source Information:")
                    st.write(metadata_summary)

                # Update chat history
                st.session_state.chat_history.append({"query": query, "response": answer + ' ' + metadata_summary})

            
    # Display Chat History (Toggleable)
    if st.session_state.show_chat_history:
        st.subheader("💬 Chat History")
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.markdown(f"**You:** {chat['query']}")
                st.markdown(f"**Assistant:** {chat['response']}")
                st.markdown("---")
        else:
            st.write("No chat history yet.")
# --------------------------- CSV Agent Tab ---------------------------
elif active_tab == "📊 CSV Agent":
    st.subheader("CSV Agent")
    st.markdown("This agent handles CSV files. Analyze and visualize data, or ask questions about your dataset.")
    
    uploaded_csv_files = [f for f in os.listdir(UPLOADED_FILES_DIR) if f.endswith(".csv")]
    if uploaded_csv_files:
        selected_file = st.sidebar.selectbox("Select a CSV file to analyze:", uploaded_csv_files)
        if selected_file:
            df = pd.read_csv(os.path.join(UPLOADED_FILES_DIR, selected_file))
            st.session_state.uploaded_df = df
            
            # Display DataFrame with vertical scroll
            st.write("### Dataframe Preview:")
            st.dataframe(df, height=200)

            # Question field for querying the dataframe "zero-shot-react-description",  'tool-calling', 'openai-tools', 'openai-functions', or 'zero-shot-react-description'
            
            agent = create_pandas_dataframe_agent(llm, df, agent_type='openai-tools', verbose=True, allow_dangerous_code=True)
            agent_query = st.text_area("🧐 Ask something about the dataframe:")
            
            if agent_query:
                with st.spinner("Processing your query..."):
                    try:
                        # Check for visualization keywords
                        if any(kw in agent_query.lower() for kw in ["plot", "chart", "scatter", "line", "bar", "visualize", ]):
                            agent_response = agent.run(agent_query)
                            st.write(agent_response)
                           
                            plt = agent.tools[0].locals.get('plt')
                            if plt:
                                st.pyplot(plt.gcf()) 
                            
                        else:
                            # Fallback to text-based query response
                            result = agent.run(agent_query)
                            st.write("### 🤖 Agent's Answer:")
                            st.write(result)

                    except Exception as e:
                        st.error(f"Error processing your query: {e}")

            
    else:
        st.write("No CSV files available. Please upload CSV files.")

# --------------------------- General Agent Tab ---------------------------
elif active_tab == "⚙️ General Agent":
    st.subheader("General Agent")
    st.markdown("Use this tab for other types of files or general data processing tasks.")
