# LLM RO-Crate Experiments

This repository contains experiments with Large Language Models (LLMs) for describing and working with RO-Crate (Research Object Crate) manifests.

## Overview

RO-Crate is a lightweight approach to packaging research data with their metadata, based on schema.org annotations in JSON-LD format. This project explores how LLMs can:

- Analyze RO-Crate manifests
- Generate human-readable descriptions of research objects
- Extract key information from crate metadata
- Validate and suggest improvements to RO-Crate structures

## Project Structure

- `token_length.py` - Utilities for analyzing token lengths in LLM inputs
- `examples/` - Sample RO-Crate manifests for testing
- `experiments/` - LLM experiment scripts and results
- `utils/` - Helper functions for RO-Crate processing

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your API keys by creating an `api_keys.json` file:
   ```bash
   cp api_keys.json.template api_keys.json
   # Then edit api_keys.json with your actual API keys
   ```
   
   The file should look like:
   ```json
   {
     "openai_api_key": "sk-proj-your_actual_openai_key_here",
     "anthropic_api_key": "your_actual_anthropic_key_here"
   }
   ```
   **Note:** This file is automatically excluded from git to keep your keys secure.

3. Test your setup:
   ```bash
   python test_setup.py
   ```

4. Explore the example RO-Crate manifests in the `examples/` directory

5. Run experiments using the scripts in the `experiments/` directory:
   ```bash
   python experiments/describe_rocrates.py
   ```
   
   This will analyze all RO-Crate manifests in the `examples/` directory and automatically save:
   - Individual description files in `outputs/`
   - A combined analysis report with timestamp
   - Summary statistics of successful/failed analyses

## RO-Crate Resources

- [RO-Crate Specification](https://www.researchobject.org/ro-crate/)
- [RO-Crate Python Library](https://github.com/ResearchObject/ro-crate-py)
- [Schema.org Documentation](https://schema.org/)

## Contributing

Feel free to add new experiments, improve existing code, or contribute example RO-Crate manifests.
