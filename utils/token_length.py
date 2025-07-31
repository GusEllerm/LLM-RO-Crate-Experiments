"""
Utility functions for working with token lengths and LLM inputs.
"""

import tiktoken
from typing import List, Dict, Any, Optional


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string for a given model.
    
    Args:
        text: The text to count tokens for
        model: The model name to use for tokenization
        
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to a default encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def truncate_text_to_tokens(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """
    Truncate text to fit within a maximum token count.
    
    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed
        model: The model name to use for tokenization
        
    Returns:
        Truncated text that fits within the token limit
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def estimate_prompt_tokens(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate the total token count for a list of chat messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name to use for tokenization
        
    Returns:
        Estimated total token count
    """
    total_tokens = 0
    
    for message in messages:
        # Add tokens for the message content
        content = message.get('content', '')
        total_tokens += count_tokens(content, model)
        
        # Add overhead tokens for message formatting
        # This is an approximation based on OpenAI's token counting
        total_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        
        if message.get('name'):
            total_tokens += count_tokens(message['name'], model)
    
    total_tokens += 2  # Every reply is primed with <im_start>assistant
    
    return total_tokens


def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int = 0, 
                        model: str = "gpt-3.5-turbo") -> List[str]:
    """
    Split text into chunks based on token count with optional overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
        model: The model name to use for tokenization
        
    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
            
        start = end - overlap
    
    return chunks


def analyze_token_distribution(texts: List[str], model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """
    Analyze token distribution across a list of texts.
    
    Args:
        texts: List of text strings to analyze
        model: The model name to use for tokenization
        
    Returns:
        Dictionary with token statistics
    """
    token_counts = [count_tokens(text, model) for text in texts]
    
    if not token_counts:
        return {}
    
    return {
        'total_texts': len(texts),
        'total_tokens': sum(token_counts),
        'average_tokens': sum(token_counts) / len(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts),
        'median_tokens': sorted(token_counts)[len(token_counts) // 2],
        'token_counts': token_counts
    }


def optimize_rocrate_for_llm(rocrate_text: str, max_tokens: int = 4000, 
                           model: str = "gpt-3.5-turbo") -> str:
    """
    Optimize RO-Crate text for LLM processing by truncating if necessary.
    
    Args:
        rocrate_text: The RO-Crate text to optimize
        max_tokens: Maximum tokens allowed
        model: The model name to use for tokenization
        
    Returns:
        Optimized text suitable for LLM processing
    """
    current_tokens = count_tokens(rocrate_text, model)
    
    if current_tokens <= max_tokens:
        return rocrate_text
    
    # If too long, try to intelligently truncate
    lines = rocrate_text.split('\n')
    
    # Keep important lines (those with key information)
    important_lines = []
    other_lines = []
    
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in [
            'dataset name:', 'description:', 'keywords:', 'creators:', 
            'published:', 'license:', 'number of files:'
        ]):
            important_lines.append(line)
        else:
            other_lines.append(line)
    
    # Start with important lines
    result_lines = important_lines[:]
    
    # Add other lines until we hit the token limit
    for line in other_lines:
        test_text = '\n'.join(result_lines + [line])
        if count_tokens(test_text, model) > max_tokens:
            break
        result_lines.append(line)
    
    optimized_text = '\n'.join(result_lines)
    
    # If still too long, truncate
    if count_tokens(optimized_text, model) > max_tokens:
        optimized_text = truncate_text_to_tokens(optimized_text, max_tokens, model)
    
    return optimized_text


class TokenBudgetManager:
    """Manage token budgets for complex LLM interactions."""
    
    def __init__(self, total_budget: int, model: str = "gpt-3.5-turbo"):
        """
        Initialize the token budget manager.
        
        Args:
            total_budget: Total token budget for the interaction
            model: Model name for tokenization
        """
        self.total_budget = total_budget
        self.model = model
        self.used_tokens = 0
        self.allocations = {}
    
    def allocate(self, component: str, tokens: int) -> bool:
        """
        Allocate tokens to a component.
        
        Args:
            component: Name of the component
            tokens: Number of tokens to allocate
            
        Returns:
            True if allocation successful, False if exceeds budget
        """
        if self.used_tokens + tokens > self.total_budget:
            return False
        
        self.allocations[component] = tokens
        self.used_tokens += tokens
        return True
    
    def get_remaining(self) -> int:
        """Get remaining tokens in the budget."""
        return self.total_budget - self.used_tokens
    
    def get_allocation(self, component: str) -> int:
        """Get allocated tokens for a component."""
        return self.allocations.get(component, 0)
    
    def can_fit(self, text: str) -> bool:
        """Check if text fits in remaining budget."""
        tokens_needed = count_tokens(text, self.model)
        return tokens_needed <= self.get_remaining()
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get a summary of the budget allocation."""
        return {
            'total_budget': self.total_budget,
            'used_tokens': self.used_tokens,
            'remaining_tokens': self.get_remaining(),
            'utilization_percent': (self.used_tokens / self.total_budget) * 100,
            'allocations': self.allocations.copy()
        }
