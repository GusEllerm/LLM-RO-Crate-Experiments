"""
Experiment: LLM-based RO-Crate Description Generation

This script demonstrates how to use various LLMs to generate human-readable
descriptions of RO-Crate manifests.
"""

import json
import os
from typing import Dict, Any, List
import openai
from pathlib import Path
import sys
from datetime import datetime

# Add the parent directory to the path to import from utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.token_length import count_tokens, optimize_rocrate_for_llm


def load_rocrate_manifest(filepath: str) -> Dict[str, Any]:
    """Load an RO-Crate manifest from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_key_info(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key information from an RO-Crate manifest."""
    graph = manifest.get('@graph', [])
    
    # Find the root dataset
    root_dataset = None
    for item in graph:
        if item.get('@id') == './' and item.get('@type') == 'Dataset':
            root_dataset = item
            break
    
    if not root_dataset:
        return {}
    
    return {
        'name': root_dataset.get('name', 'Unnamed Dataset'),
        'description': root_dataset.get('description', 'No description provided'),
        'creator': root_dataset.get('creator', []),
        'keywords': root_dataset.get('keywords', []),
        'license': root_dataset.get('license', {}),
        'datePublished': root_dataset.get('datePublished', 'Unknown'),
        'files_count': len([item for item in graph if item.get('@type') == 'File']),
        'hasPart': root_dataset.get('hasPart', [])
    }


def generate_description_prompt(key_info: Dict[str, Any], model: str = "gpt-4o") -> str:
    """Generate a prompt for LLM to describe the RO-Crate."""
    prompt = f"""
Please provide a clear, human-readable description of this research object based on the following metadata:

Title: {key_info.get('name', 'Unknown')}
Description: {key_info.get('description', 'No description')}
Publication Date: {key_info.get('datePublished', 'Unknown')}
Number of Files: {key_info.get('files_count', 0)}
Keywords: {', '.join(key_info.get('keywords', []))}
License: {key_info.get('license', {}).get('@id', 'Not specified')}

Creator(s):
{format_creators(key_info.get('creator', []))}

Files included:
{format_files(key_info.get('hasPart', []))}

Please write a comprehensive summary that would help a researcher understand what this research object contains and its potential value for their work.
"""
    
    # Check token count and optimize if necessary
    token_count = count_tokens(prompt, model)
    max_tokens = 120000 if model in ["gpt-4o", "gpt-4-turbo"] else 15000  # Leave room for response
    
    if token_count > max_tokens:
        print(f"⚠️  Prompt too long ({token_count} tokens), optimizing...")
        prompt = optimize_rocrate_for_llm(prompt, max_tokens, model)
        new_token_count = count_tokens(prompt, model)
        print(f"   Reduced to {new_token_count} tokens")
    
    return prompt


def format_creators(creators: List[Dict[str, Any]]) -> str:
    """Format creator information for the prompt."""
    if not creators:
        return "- Not specified"
    
    if isinstance(creators, dict):
        creators = [creators]
    
    formatted = []
    for creator in creators:
        name = creator.get('name', 'Unknown')
        affiliation = creator.get('affiliation', {})
        if isinstance(affiliation, dict):
            affiliation_name = affiliation.get('name', '')
            if affiliation_name:
                formatted.append(f"- {name} ({affiliation_name})")
            else:
                formatted.append(f"- {name}")
        else:
            formatted.append(f"- {name}")
    
    return '\n'.join(formatted)


def format_files(files: List[Dict[str, Any]]) -> str:
    """Format file information for the prompt."""
    if not files:
        return "- No files specified"
    
    formatted = []
    for file_ref in files:
        file_id = file_ref.get('@id', 'Unknown file')
        formatted.append(f"- {file_id}")
    
    return '\n'.join(formatted)


def save_description_to_file(filename: str, description: str, output_dir: Path) -> str:
    """Save a single description to a file."""
    output_dir.mkdir(exist_ok=True)
    
    # Create filename based on the RO-Crate name
    safe_filename = filename.replace('.json', '').replace(' ', '_').replace('/', '_')
    output_file = output_dir / f"{safe_filename}_description.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"RO-Crate Description: {filename}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(description)
        f.write("\n")
    
    return str(output_file)


def save_combined_report(results: List[Dict[str, str]], output_dir: Path, model: str) -> str:
    """Save a combined report of all descriptions."""
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f"rocrate_analysis_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RO-Crate Analysis Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model used: {model}\n")
        f.write(f"Total files analyzed: {len(results)}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{i}. Analysis of: {result['filename']}\n")
            f.write("-" * 60 + "\n")
            f.write(result['description'])
            f.write("\n\n")
        
        # Add summary
        f.write("\n" + "=" * 50 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 50 + "\n")
        
        successful = len([r for r in results if not r['description'].startswith('Error')])
        failed = len(results) - successful
        
        f.write(f"Successfully analyzed: {successful}/{len(results)} files\n")
        if failed > 0:
            f.write(f"Failed to analyze: {failed} files\n")
            f.write("\nFailed files:\n")
            for result in results:
                if result['description'].startswith('Error'):
                    f.write(f"- {result['filename']}: {result['description'][:100]}...\n")
    
    return str(report_file)


def analyze_rocrate_with_llm(manifest_path: str, model: str = "gpt-4o") -> str:
    """Analyze an RO-Crate manifest using an LLM."""
    # Load API key from api_keys.json
    try:
        with open('api_keys.json', 'r') as f:
            keys = json.load(f)
            api_key = keys.get('openai_api_key')
            if not api_key:
                return "Error: openai_api_key not found in api_keys.json"
    except FileNotFoundError:
        return "Error: api_keys.json not found. Please create it with your OpenAI API key."
    except json.JSONDecodeError:
        return "Error: Invalid JSON in api_keys.json"
    
    # Load and extract information
    manifest = load_rocrate_manifest(manifest_path)
    key_info = extract_key_info(manifest)
    
    # Generate prompt with optimization
    prompt = generate_description_prompt(key_info, model)
    
    # Call LLM
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research data specialist who helps researchers understand and discover relevant datasets and research objects."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


def main():
    """Run the experiment on sample RO-Crate manifests."""
    # Configuration
    model = "gpt-4o"  # Change this if needed
    save_output = True  # Set to False if you don't want to save files
    
    # Setup directories
    examples_dir = Path(__file__).parent.parent / "examples"
    output_dir = Path(__file__).parent.parent / "outputs"
    
    print(f"🤖 Using model: {model}")
    print(f"📁 Analyzing RO-Crate files in: {examples_dir}")
    if save_output:
        print(f"💾 Saving outputs to: {output_dir}")
    
    results = []
    
    for manifest_file in examples_dir.glob("*.json"):
        print(f"\n{'='*60}")
        print(f"Analyzing: {manifest_file.name}")
        print(f"{'='*60}")
        
        try:
            description = analyze_rocrate_with_llm(str(manifest_file), model)
            print(description)
            
            # Store result for saving
            results.append({
                'filename': manifest_file.name,
                'description': description
            })
            
            # Save individual file if requested
            if save_output:
                output_file = save_description_to_file(manifest_file.name, description, output_dir)
                print(f"\n💾 Saved to: {output_file}")
                
        except Exception as e:
            error_msg = f"Error processing {manifest_file.name}: {str(e)}"
            print(error_msg)
            results.append({
                'filename': manifest_file.name,
                'description': error_msg
            })
    
    # Save combined report
    if save_output and results:
        report_file = save_combined_report(results, output_dir, model)
        print(f"\n📋 Combined report saved to: {report_file}")
        
        # Print summary
        successful = len([r for r in results if not r['description'].startswith('Error')])
        print(f"\n📊 Summary: {successful}/{len(results)} files successfully analyzed")
    
    print(f"\n✅ Analysis complete!")
    if save_output:
        print(f"📁 All outputs saved in: {output_dir}")
    else:
        print("💡 To save outputs, set save_output=True in the main() function")


if __name__ == "__main__":
    main()
