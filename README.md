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

2. Explore the example RO-Crate manifests in the `examples/` directory

3. Run experiments using the scripts in the `experiments/` directory

## RO-Crate Resources

- [RO-Crate Specification](https://www.researchobject.org/ro-crate/)
- [RO-Crate Python Library](https://github.com/ResearchObject/ro-crate-py)
- [Schema.org Documentation](https://schema.org/)

## Contributing

Feel free to add new experiments, improve existing code, or contribute example RO-Crate manifests.
