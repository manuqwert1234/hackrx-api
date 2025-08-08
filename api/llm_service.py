import os
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMService:
    """
    Handles interactions with the Gemini LLM for query processing and response generation.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the LLM service with API key.
        
        Args:
            api_key: Gemini API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Please set GEMINI_API_KEY environment variable.")
            
        # Configure the Gemini client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Default generation config
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    
    async def extract_query_intent(self, query: str, context: str = None) -> Dict[str, Any]:
        """
        Extract the intent and key information from a user query.
        
        Args:
            query: The user's natural language query
            context: Optional context about the document or domain
            
        Returns:
            Dict containing extracted intent and entities
        """
        prompt = f"""
        Analyze the following query and extract its intent and key information.
        
        Query: {query}
        
        Context: {context or 'No additional context provided.'}
        
        Provide the response in JSON format with the following structure:
        {{
            "intent": "The main intent or purpose of the query",
            "entities": ["list", "of", "key", "entities", "or", "concepts"],
            "filters": {{
                "document_section": "specific section being referenced if any",
                "time_period": "any time period mentioned",
                "conditions": ["list", "of", "conditions", "or", "restrictions"]
            }},
            "requires_comparison": true/false,
            "is_quantitative": true/false
        }}
        """
        
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )
            
            # Parse the response
            result = self._parse_json_response(response.text)
            return result
            
        except Exception as e:
            logger.error(f"Error extracting query intent: {str(e)}")
            return {
                "intent": "unknown",
                "entities": [],
                "filters": {},
                "requires_comparison": False,
                "is_quantitative": False
            }
    
    async def generate_answer(self, query: str, context: List[str], **kwargs) -> Dict[str, Any]:
        """
        Generate an answer to a query based on the provided context.
        
        Args:
            query: The user's question
            context: List of relevant context passages
            
        Returns:
            Dict containing the generated answer and supporting evidence
        """
        # Format the context
        context_str = "\n\n".join([f"[Context {i+1}]\n{text}" for i, text in enumerate(context)])
        
        prompt = f"""
        You are an AI assistant that answers questions about documents with high accuracy.
        Use the following context to answer the question. If the answer isn't in the context, 
        say you don't know. Be concise and factual.
        
        Question: {query}
        
        Context:
        {context_str}
        
        Provide your answer in JSON format with this structure:
        {{
            "answer": "The direct answer to the question",
            "confidence": 0.0-1.0,
            "evidence": ["list", "of", "supporting", "evidence", "passages"],
            "explanation": "Brief explanation of how you arrived at this answer"
        }}
        """
        
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )
            
            # Parse the response
            result = self._parse_json_response(response.text)
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": "I'm sorry, I encountered an error processing your request.",
                "confidence": 0.0,
                "evidence": [],
                "explanation": "An error occurred while generating the response."
            }
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON response from the LLM, handling common formatting issues.
        
        Args:
            text: Raw text response from the LLM
            
        Returns:
            Parsed JSON as a dictionary
        """
        import json
        import re
        
        # Try to find JSON in the response (it might be wrapped in markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # Clean up common JSON formatting issues
        text = text.strip()
        if not (text.startswith('{') and text.endswith('}')):
            # If it's not valid JSON, try to extract just the JSON part
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                text = text[json_start:json_end]
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}\nText: {text}")
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
    
    async def generate_explanation(self, query: str, answer: str, context: List[str]) -> str:
        """
        Generate a human-readable explanation for how the answer was derived.
        
        Args:
            query: The original question
            answer: The generated answer
            context: List of relevant context passages
            
        Returns:
            A natural language explanation
        """
        context_str = "\n\n".join([f"- {text}" for text in context])
        
        prompt = f"""
        Explain how the answer was derived from the context in a clear, concise way.
        
        Question: {query}
        Answer: {answer}
        
        Context:
        {context_str}
        
        Provide a 2-3 sentence explanation of how the answer was determined from the context.
        Focus on the key pieces of information that support the answer.
        """
        
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    **self.generation_config,
                    "temperature": 0.2,  # Lower temperature for more focused explanations
                    "max_output_tokens": 256
                }
            )
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Unable to generate explanation due to an error."
