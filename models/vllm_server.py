"""vLLM server integration for GPT-OSS-20B music explanations."""

import asyncio
import logging
from typing import Dict, Any, Optional
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class MusicLLMServer:
    """LLM server for generating music recommendation explanations."""
    
    def __init__(self, model_path: str = "openai/gpt-oss-20b", 
                 gpu_memory_utilization: float = 0.8, max_model_len: int = 4096):
        """Initialize the LLM server."""
        self.model_path = model_path
        self.engine_args = AsyncEngineArgs(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="float16",
            trust_remote_code=True
        )
        self.engine: Optional[AsyncLLMEngine] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the LLM engine."""
        if not self._initialized:
            try:
                logger.info(f"Initializing vLLM engine with model: {self.model_path}")
                self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
                self._initialized = True
                logger.info("vLLM engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vLLM engine: {e}")
                raise
    
    async def explain_recommendation(self, track_info: Dict[str, Any], 
                                   user_preferences: Dict[str, Any], 
                                   confidence_score: float) -> str:
        """
        Generate explanation for a music recommendation.
        
        Args:
            track_info: Track metadata and audio features
            user_preferences: User's learned preferences
            confidence_score: AI confidence in the recommendation
            
        Returns:
            Natural language explanation
        """
        if not self._initialized:
            await self.initialize()
        
        prompt = self._build_explanation_prompt(track_info, user_preferences, confidence_score)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
            stop=["\n\n", "---"]
        )
        
        try:
            request_id = f"rec_{asyncio.current_task().get_name()}_{hash(str(track_info))}"
            results = await self.engine.generate(prompt, sampling_params, request_id)
            
            if results and len(results) > 0 and len(results[0].outputs) > 0:
                explanation = results[0].outputs[0].text.strip()
                return explanation
            else:
                return "This track matches your musical preferences based on audio features and listening history."
                
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "This track was recommended based on your musical taste and preferences."
    
    def _build_explanation_prompt(self, track_info: Dict[str, Any], 
                                user_preferences: Dict[str, Any], 
                                confidence_score: float) -> str:
        """Build the prompt for explanation generation."""
        
        track_name = track_info.get('name', 'Unknown Track')
        artist = track_info.get('artist', 'Unknown Artist')
        audio_features = track_info.get('audio_features', {})
        
        # Format audio features
        features_text = ", ".join([f"{k}: {v:.2f}" for k, v in audio_features.items()])
        
        # Format user preferences
        prefs_text = ", ".join([f"{k}: {v}" for k, v in user_preferences.items()])
        
        prompt = f"""Based on this track information and user preferences, explain why this song was recommended:

Track: {track_name} by {artist}
Audio Features: {features_text}
User Preferences: {prefs_text}
AI Confidence: {confidence_score:.2f}

Provide a concise, friendly explanation in 2-3 sentences about why this track matches the user's taste:"""
        
        return prompt
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the LLM server is healthy."""
        try:
            if not self._initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # Simple test generation
            test_prompt = "Hello, are you working?"
            sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
            request_id = "health_check"
            
            results = await self.engine.generate(test_prompt, sampling_params, request_id)
            
            if results and len(results) > 0:
                return {"status": "healthy", "healthy": True}
            else:
                return {"status": "unhealthy", "healthy": False}
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "healthy": False, "error": str(e)}


# Global LLM server instance
llm_server: Optional[MusicLLMServer] = None


async def get_llm_server() -> MusicLLMServer:
    """Get or create the global LLM server instance."""
    global llm_server
    if llm_server is None:
        llm_server = MusicLLMServer()
        await llm_server.initialize()
    return llm_server
