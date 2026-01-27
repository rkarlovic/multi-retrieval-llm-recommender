import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env file
load_dotenv()


class RecommendationGenerator:
    def __init__(self):
        # Load your HF API key from env
        self.client = InferenceClient(api_key=os.environ.get("HF_TOKEN"), bill_to="CPUI")
        self.model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        
        print(f"Using model: {self.model}")
        print(f"API provider: HuggingFace")
    
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
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            
            # Extract content from response
            if hasattr(completion, 'choices') and len(completion.choices) > 0:
                recommendation = completion.choices[0].message["content"].strip()
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
        print("PERSONALIZED RECOMMENDATION GENERATOR (HuggingFace Qwen)")
        print("="*60)
        
        generator = RecommendationGenerator()
        
        user_request = "I want a luxury hotel with spa and wellness facilities"
        print(f"\nUser request: {user_request}")
        
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
        print("1. Set HF_TOKEN environment variable: set HF_TOKEN=your_token")
        print("2. Install HuggingFace client: pip install huggingface-hub")
        print("\nNote: Requires HuggingFace API token with access to Qwen models")
