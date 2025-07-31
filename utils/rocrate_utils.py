"""
Utility functions for working with RO-Crate manifests.
"""

import json
import jsonschema
from typing import Dict, Any, List, Optional
from pathlib import Path


class ROCrateAnalyzer:
    """Analyzer for RO-Crate manifests."""
    
    def __init__(self, manifest_path: str):
        """Initialize with a path to an RO-Crate manifest."""
        self.manifest_path = Path(manifest_path)
        self.manifest = self._load_manifest()
        self.graph = self.manifest.get('@graph', [])
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load the RO-Crate manifest from file."""
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_root_dataset(self) -> Optional[Dict[str, Any]]:
        """Get the root dataset from the manifest."""
        for item in self.graph:
            if item.get('@id') == './' and 'Dataset' in item.get('@type', []):
                return item
        return None
    
    def get_files(self) -> List[Dict[str, Any]]:
        """Get all files from the manifest."""
        return [item for item in self.graph if 'File' in item.get('@type', [])]
    
    def get_people(self) -> List[Dict[str, Any]]:
        """Get all people from the manifest."""
        return [item for item in self.graph if 'Person' in item.get('@type', [])]
    
    def get_organizations(self) -> List[Dict[str, Any]]:
        """Get all organizations from the manifest."""
        return [item for item in self.graph if 'Organization' in item.get('@type', [])]
    
    def count_entities_by_type(self) -> Dict[str, int]:
        """Count entities by their @type."""
        type_counts = {}
        for item in self.graph:
            item_types = item.get('@type', [])
            if isinstance(item_types, str):
                item_types = [item_types]
            
            for type_name in item_types:
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return type_counts
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the RO-Crate."""
        root_dataset = self.get_root_dataset()
        
        return {
            'total_entities': len(self.graph),
            'files_count': len(self.get_files()),
            'people_count': len(self.get_people()),
            'organizations_count': len(self.get_organizations()),
            'has_root_dataset': root_dataset is not None,
            'entity_types': self.count_entities_by_type(),
            'name': root_dataset.get('name') if root_dataset else None,
            'description_length': len(root_dataset.get('description', '')) if root_dataset else 0,
            'keywords_count': len(root_dataset.get('keywords', [])) if root_dataset else 0,
            'has_license': bool(root_dataset.get('license')) if root_dataset else False,
            'has_date_published': bool(root_dataset.get('datePublished')) if root_dataset else False
        }
    
    def extract_text_for_llm(self) -> str:
        """Extract all text content suitable for LLM processing."""
        root_dataset = self.get_root_dataset()
        if not root_dataset:
            return "No root dataset found in this RO-Crate."
        
        text_parts = []
        
        # Basic information
        if root_dataset.get('name'):
            text_parts.append(f"Dataset Name: {root_dataset['name']}")
        
        if root_dataset.get('description'):
            text_parts.append(f"Description: {root_dataset['description']}")
        
        if root_dataset.get('keywords'):
            keywords = ', '.join(root_dataset['keywords'])
            text_parts.append(f"Keywords: {keywords}")
        
        if root_dataset.get('datePublished'):
            text_parts.append(f"Published: {root_dataset['datePublished']}")
        
        # License information
        license_info = root_dataset.get('license', {})
        if isinstance(license_info, dict) and license_info.get('@id'):
            text_parts.append(f"License: {license_info['@id']}")
        
        # Creator information
        creators = root_dataset.get('creator', [])
        if creators:
            if isinstance(creators, dict):
                creators = [creators]
            
            creator_names = []
            for creator in creators:
                name = creator.get('name', 'Unknown')
                creator_names.append(name)
            
            text_parts.append(f"Creators: {', '.join(creator_names)}")
        
        # File information
        files = self.get_files()
        if files:
            text_parts.append(f"Number of files: {len(files)}")
            
            file_descriptions = []
            for file_item in files[:10]:  # Limit to first 10 files
                file_name = file_item.get('name', file_item.get('@id', 'Unknown'))
                if file_item.get('description'):
                    file_descriptions.append(f"- {file_name}: {file_item['description']}")
                else:
                    file_descriptions.append(f"- {file_name}")
            
            if file_descriptions:
                text_parts.append("Files included:")
                text_parts.extend(file_descriptions)
                
                if len(files) > 10:
                    text_parts.append(f"... and {len(files) - 10} more files")
        
        return '\n'.join(text_parts)


def validate_rocrate_structure(manifest: Dict[str, Any]) -> List[str]:
    """Validate basic RO-Crate structure and return any issues found."""
    issues = []
    
    # Check for required @context
    if '@context' not in manifest:
        issues.append("Missing @context")
    elif manifest['@context'] != "https://w3id.org/ro/crate/1.1/context":
        issues.append("Unexpected @context value")
    
    # Check for @graph
    if '@graph' not in manifest:
        issues.append("Missing @graph")
        return issues
    
    graph = manifest['@graph']
    if not isinstance(graph, list):
        issues.append("@graph should be a list")
        return issues
    
    # Check for metadata descriptor
    metadata_descriptor = None
    for item in graph:
        if item.get('@id') == 'ro-crate-metadata.json':
            metadata_descriptor = item
            break
    
    if not metadata_descriptor:
        issues.append("Missing ro-crate-metadata.json descriptor")
    else:
        if metadata_descriptor.get('@type') != 'CreativeWork':
            issues.append("Metadata descriptor should have @type CreativeWork")
        
        conforms_to = metadata_descriptor.get('conformsTo', {})
        if conforms_to.get('@id') != 'https://w3id.org/ro/crate/1.1':
            issues.append("Metadata descriptor should conform to RO-Crate 1.1")
    
    # Check for root dataset
    root_dataset = None
    for item in graph:
        if item.get('@id') == './' and 'Dataset' in item.get('@type', []):
            root_dataset = item
            break
    
    if not root_dataset:
        issues.append("Missing root dataset (./ with @type Dataset)")
    
    return issues


def compare_rocrates(manifest1: Dict[str, Any], manifest2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two RO-Crate manifests and return differences."""
    analyzer1 = ROCrateAnalyzer.__new__(ROCrateAnalyzer)
    analyzer1.manifest = manifest1
    analyzer1.graph = manifest1.get('@graph', [])
    
    analyzer2 = ROCrateAnalyzer.__new__(ROCrateAnalyzer)
    analyzer2.manifest = manifest2
    analyzer2.graph = manifest2.get('@graph', [])
    
    stats1 = analyzer1.get_summary_stats()
    stats2 = analyzer2.get_summary_stats()
    
    return {
        'manifest1_stats': stats1,
        'manifest2_stats': stats2,
        'files_diff': stats2['files_count'] - stats1['files_count'],
        'entities_diff': stats2['total_entities'] - stats1['total_entities'],
        'people_diff': stats2['people_count'] - stats1['people_count']
    }
