import streamlit as st
import os
from langchain.text_splitter import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS



# Load environment variables
load_dotenv()

# Disable ChromaDB telemetry to reduce noise
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Set up OpenAI API key - get from Streamlit secrets
if not os.getenv("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except:
        st.error("⚠️ OpenAI API key not found. Please set it in Streamlit Cloud secrets.")
        st.stop()
        
# Streamlit page config
st.set_page_config(
    page_title="Dutch Healthcare RAG Assistant",
    layout="wide"
)

@st.cache_resource
def load_markdown_files():
    """Load all markdown files from a directory"""
    loader = DirectoryLoader(
        path="./data/english",
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    return documents

@st.cache_resource
def setup_rag_system():
    """Set up the complete RAG system - cached for performance"""
    with st.spinner("Loading knowledge base..."):
        documents = load_markdown_files()
        
        # Split documents into chunks
        text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store

        # In setup_rag_system function:
        vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
        )

        
        # Create LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return qa_chain, len(documents), len(chunks)

def ask_question(qa_chain, question, conversation_history=None):
    """Ask a question with custom healthcare funding prompt"""
    
    # Custom system prompt for healthcare funding
    system_prompt = """You are a specialized expert in Dutch healthcare innovation funding. 
    Your role is to provide clear, actionable guidance on funding options for healthcare innovations in the Netherlands.

    Guidelines:
    - Give specific, practical advice based on the provided context
    - Structure responses with clear headings when appropriate
    - Always mention relevant funding mechanisms (DBC, MSZ, facultatieve prestatie, etc.)
    - Include concrete next steps when possible
    - If information is uncertain, clearly state limitations
    - Focus on actionable outcomes rather than general information
    - Use professional but accessible language

    When the context doesn't contain sufficient information, say: "Based on the available information, I cannot provide a complete answer to this specific question. I recommend consulting with a healthcare funding expert or zorgverzekeraar for detailed guidance."
    """
    
    with st.spinner("Searching knowledge base..."):
        if conversation_history:
            enhanced_question = f"""
            {system_prompt}
            
            Previous conversation context:
            {conversation_history}
            
            Current question: {question}
            
            Please provide a comprehensive answer based on the context provided, taking into account any previous conversation.
            """
        else:
            enhanced_question = f"""
            {system_prompt}
            
            Question: {question}
            
            Please provide a comprehensive answer based on the context provided.
            """
            
        result = qa_chain.invoke({"query": enhanced_question})
        return result

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Main Streamlit App
def main():
    st.title("Dutch Healthcare Innovation Funding Assistant")
    st.markdown("### Get instant guidance on funding your healthcare innovation in the Netherlands")
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant helps healthcare innovators understand funding options 
        in the Dutch healthcare system.
        
        **Ask questions like:**
        - How do I fund my health app?
        - What is a consumer product in healthcare?
        - Can hospitals pay for my efficiency software?
        """)
        
        # Show conversation history count
        if st.session_state.conversation_history:
            st.info(f"💬 {len(st.session_state.conversation_history)} previous exchanges")
            
            # Button to clear conversation history
            if st.button("🗑️ Clear Conversation"):
                st.session_state.conversation_history = []
                st.rerun()
        
        # Example questions
        st.header("Example Questions")
        example_questions = [
            "Ik heb een gezondheidsapp ontwikkeld die mensen direct kunnen downloaden. Hoe kan ik dit laten vergoeden?",
            "Wat is een consumentenproduct in de zorg?",
            "Mijn software helpt ziekenhuizen efficiënter werken. Hoe krijg ik dit gefinancierd?",
            "Welke financieringsmogelijkheden zijn er voor een wellness product?"
        ]
        
        for i, example in enumerate(example_questions):
            if st.button(f"Example {i+1}", key=f"example_{i}", help=example):
                st.session_state.user_question = example
    
    # Initialize RAG system (cached)
    try:
        qa_chain, num_docs, num_chunks = setup_rag_system()
        
        # Show system status
        #st.success(f"✅ Knowledge base loaded: {num_docs} documents, {num_chunks} chunks")
        #st.info(f"✅ Knowledge base loaded: {num_docs} documents, {num_chunks} chunks")
        st.write(f"**✅ Knowledge base loaded:** {num_docs} documents, {num_chunks} chunks")
        
    except Exception as e:
        st.error(f"❌ Error loading system: {str(e)}")
        st.info("💡 Make sure your OpenAI API key is set and you have content in the ./content folder")
        return
    
    # Show conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        for i, exchange in enumerate(st.session_state.conversation_history):
            with st.expander(f"Exchange {i+1}: {exchange['question'][:50]}..."):
                st.markdown(f"**Question:** {exchange['question']}")
                st.markdown(f"**Answer:** {exchange['answer']}")
    
    # Question input
    st.markdown("---")
    
    # Check if example question was clicked
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    
    question = st.text_area(
        "Ask your question in Dutch or English:",
        value=st.session_state.user_question,
        placeholder="Ask your question... (Follow-ups like 'Can you explain that more?' will reference our previous conversation)",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("Ask Question", type="primary")
    
    with col2:
        if st.button("Clear"):
            st.session_state.user_question = ""
            st.rerun()
    
    # Process question
    if ask_button and question.strip():
        # Add question to conversation history first
        st.session_state.conversation_history.append({
            'question': question,
            'answer': ''  # Will be filled after getting response
        })
        
        st.markdown("---")
        st.subheader("Answer")
        
        try:
            # Build conversation context from history
            context = ""
            if len(st.session_state.conversation_history) > 1:
                # Use last 2 exchanges for context (excluding current question)
                recent_history = st.session_state.conversation_history[-3:-1]  # Exclude current
                for exchange in recent_history:
                    if exchange['answer']:  # Only include completed exchanges
                        context += f"Previous Q: {exchange['question']}\n"
                        context += f"Previous A: {exchange['answer']}\n\n"
            
            # Ask question with context
            result = ask_question(qa_chain, question, context if context else None)
            
            # Display answer
            st.markdown(result['result'])
            
            # Update the conversation history with the answer
            st.session_state.conversation_history[-1]['answer'] = result['result']
            
            # Show sources in expandable section
            with st.expander(f"📚 Sources ({len(result['source_documents'])} chunks used)"):
                for i, doc in enumerate(result['source_documents']):
                    source_file = doc.metadata.get('source', 'Unknown')
                    st.markdown(f"**Source {i+1}:** `{source_file}`")
                    st.markdown(f"```\n{doc.page_content[:300]}...\n```")
            
            # Clear the input for next question
            st.session_state.user_question = ""
                    
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            st.info("💡 This might be due to OpenAI API limits or connection issues")
            # Remove the incomplete entry from history
            if st.session_state.conversation_history and not st.session_state.conversation_history[-1]['answer']:
                st.session_state.conversation_history.pop()
    
    elif ask_button:
        st.warning("⚠️ Please enter a question first")
    
    # Footer
    st.markdown("---")
    st.markdown("*Developed by CbusineZ")

if __name__ == "__main__":
    main()
