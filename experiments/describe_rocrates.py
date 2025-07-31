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


def generate_description_prompt(key_info: Dict[str, Any]) -> str:
    """Generate a prompt for LLM to describe the RO-Crate."""
    return f"""
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


def analyze_rocrate_with_llm(manifest_path: str, model: str = "gpt-3.5-turbo") -> str:
    """Analyze an RO-Crate manifest using an LLM."""
    # Load and extract information
    manifest = load_rocrate_manifest(manifest_path)
    key_info = extract_key_info(manifest)
    
    # Generate prompt
    prompt = generate_description_prompt(key_info)
    
    # Call LLM (requires OpenAI API key)
    try:
        client = openai.OpenAI()
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
    examples_dir = Path(__file__).parent.parent / "examples"
    
    for manifest_file in examples_dir.glob("*.json"):
        print(f"\n{'='*60}")
        print(f"Analyzing: {manifest_file.name}")
        print(f"{'='*60}")
        
        try:
            description = analyze_rocrate_with_llm(str(manifest_file))
            print(description)
        except Exception as e:
            print(f"Error processing {manifest_file.name}: {str(e)}")


if __name__ == "__main__":
    main()
