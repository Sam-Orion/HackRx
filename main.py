# This file contains the FastAPI application that serves as the main entry point for the system.

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
import os
from typing import List, Optional
from agents import QueryParsingAgent, ClauseRetrievalAgent, DecisionMakingAgent, JSONFormattingAgent
from ingest import DocumentIngestionAgent

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Insurance Claim Processing Agentic System",
    description="An AI-powered system to process insurance claims from policy documents.",
    version="1.0.0"
)

# --- Pydantic Models for API ---
class ParsedQuery(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration_months: Optional[int] = None

class Justification(BaseModel):
    clause: str
    decision_reasoning: str

class ClaimResponse(BaseModel):
    decision: str
    amount: Optional[float] = None
    justification: List[Justification]

# --- Agent Initialization ---
# It's a good practice to initialize agents once and reuse them.
query_parsing_agent = QueryParsingAgent()
clause_retrieval_agent = ClauseRetrievalAgent()
decision_making_agent = DecisionMakingAgent()
json_formatting_agent = JSONFormattingAgent()
document_ingestion_agent = DocumentIngestionAgent()

# --- API Endpoints ---
@app.post("/upload_documents/", summary="Upload and process policy documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Uploads one or more policy documents (PDFs) and processes them into a vector store.
    """
    file_paths = []
    for file in files:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        file_paths.append(file_path)

    document_ingestion_agent.ingest(file_paths)

    # Clean up temporary files
    for file_path in file_paths:
        os.remove(file_path)

    return {"message": "Documents uploaded and processed successfully."}

@app.post("/process_claim/", response_model=ClaimResponse, summary="Process a natural language insurance claim")
async def process_claim(query: str = Form(...)):
    """
    Processes a natural language query against the uploaded policy documents and returns a structured claim decision.
    """
    # 1. Parse the query
    parsed_query = query_parsing_agent.parse(query)

    # 2. Retrieve relevant clauses
    retrieved_clauses = clause_retrieval_agent.retrieve(str(parsed_query))

    # 3. Make a decision
    decision_and_justification = decision_making_agent.decide(parsed_query, retrieved_clauses)

    # 4. Format the response
    formatted_response = json_formatting_agent.format(decision_and_justification)

    return formatted_response

# --- Main execution block ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)