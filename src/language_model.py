from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from typing import Any


class LanguageModel(object):
    def __init__(self, model_name: str, initial_prompt: str, temperature: float = 0.1):
        assert model_name is not None, "Model name cannot be None"
        assert initial_prompt is not None, "Initial prompt cannot be None"

        try:
            _llm = ChatOllama(model = model_name, temperature = temperature)
            _prompt = ChatPromptTemplate.from_messages([
                ("system", initial_prompt + "\n\nContexto:\n{context}"),
                ("human", "{input}")
            ])
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

        self.llm: ChatOllama = _llm
        self.prompt: ChatPromptTemplate = _prompt
        self.docs_chain: Runnable[dict[str, Any], Any] = create_stuff_documents_chain(llm = self.llm, 
                                                                                      prompt = self.prompt)
        self.rag_chain: Runnable = None

    
    def define_rag_chain(self, retriever: EnsembleRetriever) -> None:
        self.rag_chain = create_retrieval_chain(retriever = retriever, combine_docs_chain = self.docs_chain)

    
    def generate_response(self, pregunta: str) -> str:
        if self.rag_chain is None:
            raise ValueError("RAG chain is not defined. Please call define_rag_chain() first.")
        
        response = self.rag_chain.invoke({"input": pregunta})

        return response["answer"]