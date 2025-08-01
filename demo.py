
import os
from langchain.text_splitter import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "Key"

# Document loader 

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


# Set up RAG

def setup_rag_system():
    """Set up the complete RAG system"""
    print("Loading markdown files...")
    documents = load_markdown_files()
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
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
    
    return qa_chain

def ask_question(qa_chain, question):
    """Ask a question and get answer with sources"""
    print(f"\nQuestion: {question}")
    result = qa_chain({"query": question})
    
    print(f"Answer: {result['result']}")
    print(f"Number of sources used: {len(result['source_documents'])}")
    
    # Show source filenames
    sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
    print(f"Sources: {', '.join(set(sources))}")
    print("-" * 80)
    
    return result


def main():
    """Main demo function"""
    print("=== Dutch Healthcare RAG Demo ===")
    
    # Setup RAG system
    qa_chain = setup_rag_system()
    
    # Test questions
    test_questions = [
        "Ik heb een gezondheidsapp ontwikkeld die mensen direct kunnen downloaden. Hoe kan ik dit laten vergoeden door zorgverzekeraars?",
        "Ik heb software ontwikkeld die ziekenhuizen helpt om patiëntgegevens efficiënter te beheren. Hoe krijg ik dit gefinancierd?",
        "Wat is een consumentenproduct in de zorg?",
        "Mijn innovatie is een wellness product dat consumenten zelf kunnen kopen. Welke financieringsmogelijkheden heb ik?"
    ]
    
    # Ask each question - testing
    # for question in test_questions:
    #     ask_question(qa_chain, question)
    
    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Ask your own questions (type 'quit' to exit):")
    
    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'stop']:
            break
        if user_question:
            ask_question(qa_chain, user_question)

if __name__ == "__main__":
    main()
