"""
AI Chat endpoint with RAG (Retrieval-Augmented Generation) using Ollama and embeddings.
Handles student questions with context from course materials.
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Any, Optional
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDINGS_API_URL = os.getenv("EMBEDDINGS_API_URL", "http://localhost:8001")
TICKETS_API_URL = os.getenv("TICKETS_API_URL", "http://localhost:8000")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b")
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "6"))

# System and User prompt templates
SYSTEM_PROMPT = """SYSTEM:
You are the upGrad AI Mentor. Behavior rules:
- Use ONLY the provided CONTEXT chunks (do not hallucinate). If the model lacks enough info, respond with "I don't know. Creating internal ticket." and set "escalate": true.
- Write concise explanations (max 300 words) with an example if applicable.
- Provide 2 short follow-up questions to clarify student needs.
- Always include "SOURCES" with each cited chunk as DOC:{doc_id}:{chunk_id}.
- Detect language and respond in the student's language (Hindi/Marathi/English).
- If the query is administrative (payments, refunds), do not answer — escalate to human.
Safety: block medical/legal/financial advice — escalate."""

USER_PROMPT_TEMPLATE = """USER_PROMPT:
Context: {course_name} | module: {module_name} | user_lang: {user_lang} | user_id_hash: {user_hash}
CONTEXT:
{context_chunks}

INSTRUCTION:
Student question: "{student_question}"

Task:
1. Answer concisely with step-by-step explanation and at least one small example or micro-exercise.
2. Add a short 1-2 question micro-quiz to practice (if relevant).
3. Return sources as a list of DOC:{doc_id}:{chunk_id}.
4. Provide two follow-up clarification questions.
5. Output final block as valid JSON on the last line: 
{"answer": "...", "sources":[{"doc":"doc_id","chunk":"chunk_id","score":0.92}], "followups":["...","..."], "escalate": false}"""

# Pydantic models
class AIAskRequest(BaseModel):
    user_id_hash: str = Field(..., description="Hashed user identifier")
    course_id: str = Field(..., description="Course identifier")
    module_id: str = Field(..., description="Module identifier")
    question: str = Field(..., description="Student's question")
    lang: str = Field(default="English", description="Preferred language (English/Hindi/Marathi)")

class SourceInfo(BaseModel):
    doc: str
    chunk: str
    score: float

class AIAskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    followups: List[str]
    escalate: bool
    ticket_id: Optional[str] = None

class TicketRequest(BaseModel):
    title: str
    description: str
    status: str = "open"
    priority: str = "medium"
    created_by: str

# Initialize FastAPI app
app = FastAPI(title="AI Chat API", version="1.0.0")

class AIService:
    """Service class for handling AI chat with RAG."""
    
    def __init__(self):
        self.ollama_url = OLLAMA_BASE_URL
        self.embeddings_url = EMBEDDINGS_API_URL
        self.tickets_url = TICKETS_API_URL
        self.llm_model = LLM_MODEL
        self.search_top_k = SEARCH_TOP_K
    
    async def _search_embeddings(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search for relevant context using embeddings API."""
        if k is None:
            k = self.search_top_k
        
        search_payload = {
            "query": query,
            "k": k
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.embeddings_url}/api/embeddings/search",
                    json=search_payload
                )
                
                if response.status_code != 200:
                    logger.error(f"Embeddings search failed: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Embeddings search service unavailable"
                    )
                
                result = response.json()
                return result.get("results", [])
                
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to embeddings service: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embeddings service connection failed"
            )
    
    def _format_context_chunks(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context chunks for the prompt."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            payload = result.get("payload", {})
            text = payload.get("text", "")
            doc_id = payload.get("original_doc_id", payload.get("doc_id", "unknown"))
            chunk_id = payload.get("chunk_id", f"chunk_{i}")
            score = result.get("score", 0.0)
            
            # Create metadata string
            meta = f"DOC:{doc_id}:{chunk_id} (score:{score:.2f})"
            context_parts.append(f"{meta} - {text}")
        
        return "\n".join(context_parts)
    
    def _build_full_prompt(self, request: AIAskRequest, context_chunks: str) -> str:
        """Build the complete prompt with system and user parts."""
        # Format user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            course_name=request.course_id,
            module_name=request.module_id,
            user_lang=request.lang,
            user_hash=request.user_id_hash,
            context_chunks=context_chunks,
            student_question=request.question
        )
        
        # Combine system and user prompts
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        return full_prompt
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using Ollama LLM."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,  # Set to True for streaming if needed
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }
                
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama generation failed: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="LLM generation service unavailable"
                    )
                
                result = response.json()
                return result.get("response", "")
                
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to Ollama service: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service connection failed"
            )
    
    def _parse_json_response(self, llm_response: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse the JSON response from LLM output."""
        # Try to find JSON at the end of the response
        json_pattern = r'\{[^{}]*"answer"[^{}]*"sources"[^{}]*"followups"[^{}]*"escalate"[^{}]*\}'
        
        # Look for JSON in the last few lines
        lines = llm_response.strip().split('\n')
        json_text = None
        
        # Try to find JSON in reverse order (from end)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    json.loads(line)
                    json_text = line
                    break
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found in lines, try regex search
        if not json_text:
            matches = re.findall(json_pattern, llm_response, re.DOTALL)
            if matches:
                json_text = matches[-1]  # Take the last match
        
        if json_text:
            try:
                parsed = json.loads(json_text)
                
                # Validate required fields
                required_fields = ["answer", "sources", "followups", "escalate"]
                if all(field in parsed for field in required_fields):
                    # Ensure sources have proper format
                    formatted_sources = []
                    for source in parsed.get("sources", []):
                        if isinstance(source, dict) and "doc" in source and "chunk" in source:
                            formatted_sources.append({
                                "doc": source["doc"],
                                "chunk": source["chunk"],
                                "score": source.get("score", 0.0)
                            })
                    
                    parsed["sources"] = formatted_sources
                    return parsed
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
        
        # Fallback: create response from search results
        logger.warning("Could not parse structured JSON from LLM response, creating fallback")
        
        # Extract answer (everything before potential JSON)
        answer_text = llm_response
        if json_text:
            answer_text = llm_response.replace(json_text, "").strip()
        
        # Create sources from search results
        sources = []
        for result in search_results[:3]:  # Top 3 sources
            payload = result.get("payload", {})
            doc_id = payload.get("original_doc_id", payload.get("doc_id", "unknown"))
            chunk_id = payload.get("chunk_id", "unknown")
            score = result.get("score", 0.0)
            
            sources.append({
                "doc": doc_id,
                "chunk": chunk_id,
                "score": score
            })
        
        return {
            "answer": answer_text or "I apologize, but I couldn't generate a proper response. Please try rephrasing your question.",
            "sources": sources,
            "followups": [
                "Could you provide more specific details about what you'd like to learn?",
                "Would you like me to explain any particular concept in more depth?"
            ],
            "escalate": len(search_results) == 0  # Escalate if no context found
        }
    
    async def _create_ticket(self, request: AIAskRequest, reason: str) -> Optional[str]:
        """Create a support ticket when escalation is needed."""
        try:
            ticket_data = {
                "title": f"AI Escalation: {request.question[:50]}...",
                "description": f"Question: {request.question}\n\nCourse: {request.course_id}\nModule: {request.module_id}\nUser: {request.user_id_hash}\nLanguage: {request.lang}\n\nReason for escalation: {reason}",
                "status": "open",
                "priority": "medium",
                "created_by": f"ai_system_{request.user_id_hash}"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.tickets_url}/api/tickets",
                    json=ticket_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ticket_id = result.get("id")
                    logger.info(f"Created escalation ticket {ticket_id} for user {request.user_id_hash}")
                    return str(ticket_id) if ticket_id else None
                else:
                    logger.error(f"Failed to create ticket: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"Error creating escalation ticket: {e}")
        
        return None
    
    def _detect_administrative_query(self, question: str) -> bool:
        """Detect if the question is administrative (payments, refunds, etc.)."""
        admin_keywords = [
            "payment", "refund", "billing", "invoice", "subscription", "cancel",
            "account", "login", "password", "registration", "enrollment",
            "certificate", "completion", "grade", "marks", "score"
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in admin_keywords)
    
    def _detect_unsafe_query(self, question: str) -> bool:
        """Detect medical/legal/financial advice queries."""
        unsafe_keywords = [
            "medical", "health", "disease", "medicine", "treatment", "diagnosis",
            "legal", "law", "lawsuit", "attorney", "court", "contract",
            "financial advice", "investment", "stock", "trading", "loan", "mortgage"
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in unsafe_keywords)
    
    async def process_question(self, request: AIAskRequest) -> AIAskResponse:
        """Process student question with RAG pipeline."""
        logger.info(f"Processing question from user {request.user_id_hash}: {request.question[:100]}...")
        
        # Check for administrative or unsafe queries
        if self._detect_administrative_query(request.question):
            logger.info("Detected administrative query, escalating to human")
            ticket_id = await self._create_ticket(request, "Administrative query")
            return AIAskResponse(
                answer="This appears to be an administrative question. I've created a support ticket for you, and our team will get back to you soon.",
                sources=[],
                followups=[
                    "Is there anything else about the course content I can help you with?",
                    "Would you like to know more about any specific topic from your modules?"
                ],
                escalate=True,
                ticket_id=ticket_id
            )
        
        if self._detect_unsafe_query(request.question):
            logger.info("Detected unsafe query, escalating to human")
            ticket_id = await self._create_ticket(request, "Unsafe query (medical/legal/financial)")
            return AIAskResponse(
                answer="I can't provide medical, legal, or financial advice. I've created a support ticket for you to get appropriate help.",
                sources=[],
                followups=[
                    "Can I help you with any course-related topics instead?",
                    "Would you like to explore the learning materials in your current module?"
                ],
                escalate=True,
                ticket_id=ticket_id
            )
        
        try:
            # Step 1: Search for relevant context
            search_results = await self._search_embeddings(request.question)
            
            if not search_results:
                logger.warning("No relevant context found for question")
                ticket_id = await self._create_ticket(request, "No relevant context found")
                return AIAskResponse(
                    answer="I don't know. Creating internal ticket.",
                    sources=[],
                    followups=[
                        "Could you try rephrasing your question?",
                        "Is there a specific topic or module you'd like to focus on?"
                    ],
                    escalate=True,
                    ticket_id=ticket_id
                )
            
            # Step 2: Format context chunks
            context_chunks = self._format_context_chunks(search_results)
            
            # Step 3: Build full prompt
            full_prompt = self._build_full_prompt(request, context_chunks)
            
            # Step 4: Generate response using LLM
            llm_response = await self._generate_response(full_prompt)
            
            # Step 5: Parse structured response
            parsed_response = self._parse_json_response(llm_response, search_results)
            
            # Step 6: Handle escalation if needed
            ticket_id = None
            if parsed_response.get("escalate", False):
                ticket_id = await self._create_ticket(request, "LLM requested escalation")
            
            return AIAskResponse(
                answer=parsed_response["answer"],
                sources=[SourceInfo(**source) for source in parsed_response["sources"]],
                followups=parsed_response["followups"],
                escalate=parsed_response["escalate"],
                ticket_id=ticket_id
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            # Create fallback ticket
            ticket_id = await self._create_ticket(request, f"System error: {str(e)}")
            return AIAskResponse(
                answer="I encountered an error while processing your question. I've created a support ticket for you.",
                sources=[],
                followups=[
                    "Please try asking your question again in a few minutes.",
                    "Is there anything else I can help you with?"
                ],
                escalate=True,
                ticket_id=ticket_id
            )

# Initialize service
ai_service = AIService()

# API Endpoints
@app.post("/api/ai/ask", response_model=AIAskResponse)
async def ai_ask(request: AIAskRequest):
    """
    AI Chat endpoint with RAG (Retrieval-Augmented Generation).
    
    Processes student questions by:
    1. Searching for relevant context using embeddings
    2. Building a prompt with system instructions and context
    3. Generating response using Ollama LLM
    4. Parsing structured JSON response
    5. Creating support tickets for escalations
    """
    logger.info(f"AI chat request from user {request.user_id_hash}")
    
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    return await ai_service.process_question(request)

@app.get("/api/ai/health")
async def health_check():
    """Health check endpoint for the AI service."""
    try:
        # Check Ollama connection
        async with httpx.AsyncClient(timeout=5.0) as client:
            ollama_response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = ollama_response.status_code == 200
        
        # Check embeddings service
        embeddings_status = False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                embeddings_response = await client.get(f"{EMBEDDINGS_API_URL}/api/embeddings/health")
                embeddings_status = embeddings_response.status_code == 200
        except Exception:
            pass
        
        return {
            "status": "healthy" if (ollama_status and embeddings_status) else "degraded",
            "services": {
                "ollama": "up" if ollama_status else "down",
                "embeddings": "up" if embeddings_status else "down"
            },
            "config": {
                "llm_model": LLM_MODEL,
                "search_top_k": SEARCH_TOP_K,
                "ollama_url": OLLAMA_BASE_URL,
                "embeddings_url": EMBEDDINGS_API_URL
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ai:app",
        host="0.0.0.0",
        port=8002,
        reload=True
    )
