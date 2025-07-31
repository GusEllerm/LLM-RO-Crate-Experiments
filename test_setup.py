"""
Simple test script to verify LLM integration with RO-Crate analysis.
"""

import json
import sys
from pathlib import Path
import openai
from utils.rocrate_utils import ROCrateAnalyzer


def load_api_key():
    """Load OpenAI API key from api_keys.json."""
    try:
        with open('api_keys.json', 'r') as f:
            keys = json.load(f)
            return keys.get('openai_api_key')
    except FileNotFoundError:
        print("Error: api_keys.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON in api_keys.json")
        return None


def test_rocrate_analysis():
    """Test RO-Crate analysis with a sample manifest."""
    print("🔍 Testing RO-Crate Analysis...")
    
    # Test with the climate dataset example
    climate_file = Path("examples/sample_climate_crate.json")
    
    if not climate_file.exists():
        print(f"❌ Sample file not found: {climate_file}")
        return False
    
    try:
        analyzer = ROCrateAnalyzer(str(climate_file))
        stats = analyzer.get_summary_stats()
        
        print(f"✅ Successfully loaded RO-Crate: {stats['name']}")
        print(f"   - Total entities: {stats['total_entities']}")
        print(f"   - Files: {stats['files_count']}")
        print(f"   - People: {stats['people_count']}")
        print(f"   - Has license: {stats['has_license']}")
        
        return True
    except Exception as e:
        print(f"❌ Error analyzing RO-Crate: {e}")
        return False


def test_llm_integration():
    """Test LLM integration with a simple request."""
    print("\n🤖 Testing LLM Integration...")
    
    api_key = load_api_key()
    if not api_key:
        print("❌ Could not load API key")
        return False
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Simple test with the climate dataset
        climate_file = Path("examples/sample_climate_crate.json")
        analyzer = ROCrateAnalyzer(str(climate_file))
        rocrate_text = analyzer.extract_text_for_llm()
        
        # Create a simple prompt
        prompt = f"""Please provide a brief, one-paragraph summary of this research dataset:

{rocrate_text}

Focus on what the dataset contains and who might find it useful."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful research data specialist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content
        print("✅ LLM Response:")
        print(f"   {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with LLM integration: {e}")
        return False


def test_token_counting():
    """Test token counting functionality."""
    print("\n📊 Testing Token Counting...")
    
    try:
        # Import here to handle missing dependencies gracefully
        from utils.token_length import count_tokens, analyze_token_distribution
        
        # Load a sample RO-Crate
        climate_file = Path("examples/sample_climate_crate.json")
        analyzer = ROCrateAnalyzer(str(climate_file))
        rocrate_text = analyzer.extract_text_for_llm()
        
        tokens = count_tokens(rocrate_text, "gpt-3.5-turbo")
        print(f"✅ Token count for climate dataset: {tokens} tokens")
        
        # Test with multiple examples
        examples = []
        for example_file in Path("examples").glob("*.json"):
            try:
                example_analyzer = ROCrateAnalyzer(str(example_file))
                example_text = example_analyzer.extract_text_for_llm()
                examples.append(example_text)
            except:
                continue
        
        if examples:
            stats = analyze_token_distribution(examples)
            print(f"   Average tokens across examples: {stats.get('average_tokens', 0):.1f}")
            print(f"   Range: {stats.get('min_tokens', 0)} - {stats.get('max_tokens', 0)} tokens")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Token counting requires tiktoken: {e}")
        return False
    except Exception as e:
        print(f"❌ Error with token counting: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running LLM RO-Crate Integration Tests\n")
    
    tests = [
        ("RO-Crate Analysis", test_rocrate_analysis),
        ("Token Counting", test_token_counting),
        ("LLM Integration", test_llm_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("📋 Test Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready for RO-Crate experiments.")
    else:
        print("\n💡 Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()
