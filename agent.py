# This module defines the different agents responsible for the various tasks in the system.

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch # Import torch

# --- Base Models for Agent Outputs ---
class ParsedQuery(BaseModel):
    age: Optional[int] = Field(None, description="The age of the person.")
    gender: Optional[str] = Field(None, description="The gender of the person.")
    procedure: Optional[str] = Field(None, description="The medical procedure or surgery.")
    location: Optional[str] = Field(None, description="The location where the procedure took place.")
    policy_duration_months: Optional[int] = Field(None, description="The duration of the insurance policy in months.")

class Justification(BaseModel):
    clause: str = Field(..., description="The specific clause from the policy document.")
    decision_reasoning: str = Field(..., description="The reasoning behind the decision based on the clause.")

class Decision(BaseModel):
    decision: str = Field(..., description="The final decision, e.g., 'approved' or 'rejected'.")
    amount: Optional[float] = Field(None, description="The payout amount, if applicable.")
    justification: List[Justification]

# --- LLM Initialization (Free, Local Model) ---
# Switched to a smaller model for better performance on consumer hardware.
model_id = "google/flan-t5-small" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Explicitly set the device to CPU to avoid potential issues with limited VRAM on the MX130.
# If you have a more powerful GPU, you can change 'cpu' to 'cuda'.
device = "cpu" 

pipe = pipeline(
    "text2text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=512,
    device=torch.device(device) # Use the specified device
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- Query Parsing Agent ---
class QueryParsingAgent:
    def __init__(self):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ParsedQuery)
        self.prompt = PromptTemplate(
            template="""
            You are an expert at parsing insurance claim queries.
            Extract the key details from the following query: "{query}"
            {format_instructions}
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def parse(self, query: str) -> ParsedQuery:
        output = self.chain.run(query)
        # Simple error handling for parsing
        try:
            return self.parser.parse(output)
        except Exception as e:
            print(f"Error parsing query output: {e}")
            # Return an empty object if parsing fails, so the pipeline can continue
            return ParsedQuery()


# --- Clause Retrieval Agent ---
class ClauseRetrievalAgent:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="/tmp/chroma_db")
        self.collection = self.client.get_or_create_collection(name="insurance_policies")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, query: str, n_results: int = 5) -> List[str]:
        if not query.strip() or query == "{}": # Avoid querying with empty strings
             return []
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []


# --- Decision Making Agent ---
class DecisionMakingAgent:
    def __init__(self):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=Decision)
        self.prompt = PromptTemplate(
            template="""
            You are an insurance claim decision expert. Based on the following parsed query and retrieved policy clauses,
            make a decision and provide a detailed justification.

            Parsed Query: {parsed_query}

            Retrieved Clauses:
            {retrieved_clauses}

            Your task is to determine if the claim should be approved or rejected and explain why, referencing the specific clauses.
            The final output should be a structured JSON.
            {format_instructions}
            """,
            input_variables=["parsed_query", "retrieved_clauses"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def decide(self, parsed_query: ParsedQuery, retrieved_clauses: List[str]) -> str:
        if not retrieved_clauses:
             # Handle cases where no clauses were found
             return '{"decision": "undetermined", "justification": [{"clause": "N/A", "decision_reasoning": "No relevant policy clauses could be found to make a decision."}]}'
        
        # Combine retrieved clauses into a single string
        clauses_str = "\n---\n".join(retrieved_clauses)
        
        # Format the parsed query for the prompt
        query_str = str(parsed_query.dict())

        return self.chain.run(parsed_query=query_str, retrieved_clauses=clauses_str)

# --- JSON Formatting Agent ---
class JSONFormattingAgent:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=Decision)

    def format(self, decision_str: str) -> Decision:
        try:
            return self.parser.parse(decision_str)
        except Exception as e:
            print(f"Error parsing decision output: {e}")
            # Provide a fallback response in case of a formatting error from the LLM
            return Decision(
                decision="error",
                justification=[Justification(
                    clause="N/A",
                    decision_reasoning=f"The model failed to produce a correctly formatted JSON response. Raw output: {decision_str}"
                )]
            )