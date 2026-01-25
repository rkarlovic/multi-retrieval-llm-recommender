import json
from litellm import completion

class RecommendationGenerator:
    def __init__(self, model=None, api_base="http://localhost:11434"):

        self.model = model or "ollama/mistral:latest"
        self.api_base = api_base
        
        print(f"Using model: {self.model}")
        print(f"API base: {self.api_base}")
    
    def generate_recommendation(self, user_request, user_context):

        context_str = json.dumps(user_context, indent=2)
        
        prompt = f"""Based on the user's request and their past experiences, generate a personalized recommendation for activities they might enjoy during their stay.

User Request: {user_request}

User Context:
{context_str}

IMPORTANT RULES:
1. Generate 100-200 words of recommendations
2. Focus on types of activities: cultural experiences, sports, gourmet dining, nature, entertainment, etc.
3. DO NOT mention any specific names of places, restaurants, companies, brands, services, products, or hotels
4. Use general categories and descriptions instead (e.g., "a local seafood restaurant" instead of "Restaurant XYZ")
5. Base recommendations on the user's past preferences and experiences

Generate the recommendation now:"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful travel recommendation assistant. Provide personalized activity suggestions based on user preferences without mentioning specific business names."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print("\nGenerating recommendation...")
        
        try:
            response = completion(
                model=self.model,
                messages=messages,
                api_base=self.api_base,
                temperature=0.7,
                max_tokens=300
            )
            
            # Extract content from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                recommendation = response.choices[0].message.content.strip()
                print("✓ Generation complete!")
                return recommendation
            else:
                raise Exception("No response content received from model")
                
        except Exception as e:
            raise Exception(f"Error generating recommendation: {str(e)}")


# Example usage
if __name__ == "__main__":
    user_context = {
        "past_stays": [
            {
                "location": "coastal area",
                "duration": "5 days",
                "activities_enjoyed": ["beach activities", "seafood dining", "sunset viewing"],
                "rating": 5
            },
            {
                "location": "mountain region",
                "duration": "3 days",
                "activities_enjoyed": ["hiking", "local cuisine", "photography"],
                "rating": 4
            }
        ],
        "preferences": {
            "activity_level": "moderate to active",
            "dining_style": "local and authentic",
            "interests": ["nature", "photography", "cultural experiences", "outdoor activities"]
        },
        "past_experiences": {
            "enjoyed": ["water sports", "hiking trails", "local markets", "traditional cuisine"],
            "avoided": ["crowded tourist spots", "fast food", "indoor shopping malls"]
        }
    }
    
    try:
        print("="*60)
        print("PERSONALIZED RECOMMENDATION GENERATOR (OLLAMA)")
        print("="*60)
        
        generator = RecommendationGenerator()
        # generator = RecommendationGenerator(model="ollama/llama3.1:8b")
        # generator = RecommendationGenerator(model="ollama/mistral:7b")
        # generator = RecommendationGenerator(model="ollama/phi3")
        # generator = RecommendationGenerator(model="ollama/gemma2")
        # generator = RecommendationGenerator(model="ollama/deepseek-r1:14b")
        
        user_request = "I want recommendations for activities during my upcoming 4-day stay"
        
        print(f"\nUser request: {user_request}")
        print("="*60)
        
        recommendation = generator.generate_recommendation(user_request, user_context)
        
        print("\n" + "="*60)
        print("GENERATED RECOMMENDATION:")
        print("="*60)
        print(recommendation)
        print("="*60)
        print(f"Word count: {len(recommendation.split())} words")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nSetup Instructions:")
        print("1. Install Ollama: https://ollama.ai/download")
        print("2. Start Ollama: ollama serve")
        print("3. Pull a model: ollama pull llama3.2")
        print("4. Install litellm: pip install litellm")
        print("\nPopular models to try:")
        print("  - ollama/llama3.2")
        print("  - ollama/llama3.1:8b")
        print("  - ollama/mistral:7b")
        print("  - ollama/phi3")
        print("  - ollama/gemma2")
        print("  - ollama/deepseek-r1:14b")