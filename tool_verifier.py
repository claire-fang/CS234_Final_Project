"""
Tool Verifier: Validates whether tool selection is correct for a task.

A verifier that outputs a binary signal (yes/no) indicating whether a chosen tool
was appropriate for the task context. This can be used independently or integrated
with PPO training.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import re


@dataclass
class VerificationResult:
    """Result of tool verification."""
    tool: str
    task: str
    is_valid: bool  # yes/no signal
    confidence: float  # 0-1 confidence
    reason: str  # Why this tool was/wasn't appropriate
    metadata: Dict = None


class SimpleToolVerifier:
    """
    Simple rule-based verifier for tool selection.
    
    Outputs: yes/no signal indicating if tool choice is appropriate.
    
    Can be extended with learned components (neural net).
    """
    
    # Tool signatures: what each tool is good for
    TOOL_SIGNATURES = {
        "calculator": {
            "keywords": ["calculate", "math", "compute", "plus", "minus", "multiply", "divide", 
                        "sum", "square", "power", "percentage", "fraction", "equation"],
            "patterns": [r"\d+\s*[\+\-\*/]", r"[\+\-\*\/]\s*\d+"],
            "description": "Arithmetic calculations"
        },
        "web_search": {
            "keywords": ["search", "find", "look up", "who", "what", "where", "when", "why",
                        "current", "latest", "recent", "news", "fact", "information"],
            "patterns": None,
            "description": "Finding information from web"
        },
        "web_fetch": {
            "keywords": ["fetch", "url", "page", "website", "content", "article", "read"],
            "patterns": [r"http[s]?://\S+"],
            "description": "Fetching content from specific URLs"
        },
        "extract": {
            "keywords": ["extract", "parse", "html", "text", "content", "from"],
            "patterns": [r"<[^>]+>"],
            "description": "Extracting content from HTML"
        },
    }
    
    def verify(self, task: str, tool: str, context: str = "") -> VerificationResult:
        """
        Verify if tool choice is appropriate.
        
        Args:
            task: Task description/query
            tool: Tool name to verify
            context: Additional context (previous steps, etc.)
        
        Returns:
            VerificationResult with yes/no signal
        """
        combined_text = f"{task} {context}".lower()
        
        if tool not in self.TOOL_SIGNATURES:
            return VerificationResult(
                tool=tool,
                task=task,
                is_valid=False,
                confidence=1.0,
                reason=f"Unknown tool: {tool}"
            )
        
        sig = self.TOOL_SIGNATURES[tool]
        
        # Check keyword match
        keyword_score = 0.0
        for keyword in sig["keywords"]:
            if keyword in combined_text:
                keyword_score += 1.0
        
        if len(sig["keywords"]) > 0:
            keyword_score /= len(sig["keywords"])
        
        # Check pattern match
        pattern_score = 0.0
        if sig["patterns"]:
            for pattern in sig["patterns"]:
                if re.search(pattern, combined_text):
                    pattern_score = 1.0
                    break
        
        # Combined score
        combined_score = 0.6 * keyword_score + 0.4 * pattern_score
        
        # Decision threshold
        is_valid = combined_score > 0.3
        confidence = min(1.0, max(0.0, combined_score))
        
        if is_valid:
            reason = f"Tool appropriate (score={combined_score:.2f}): {sig['description']}"
        else:
            reason = f"Tool not suitable (score={combined_score:.2f}): {sig['description']}"
        
        return VerificationResult(
            tool=tool,
            task=task,
            is_valid=is_valid,
            confidence=confidence,
            reason=reason
        )
    
    def verify_batch(self, task: str, tools: List[str], 
                    context: str = "") -> List[VerificationResult]:
        """Verify multiple tool choices for same task."""
        return [self.verify(task, tool, context) for tool in tools]
    
    def suggest_best_tools(self, task: str, context: str = "", 
                          top_k: int = 1) -> List[Tuple[str, float]]:
        """Suggest best tools for a task."""
        results = self.verify_batch(task, list(self.TOOL_SIGNATURES.keys()), context)
        
        # Sort by confidence
        valid_tools = [(r.tool, r.confidence) for r in results if r.is_valid]
        valid_tools.sort(key=lambda x: -x[1])
        
        return valid_tools[:top_k] if valid_tools else []


class LearnableToolVerifier:
    """
    Verifier that can be fine-tuned to learn tool correctness.
    
    Uses a simple neural network to predict tool correctness.
    Can be trained on trajectories + task outcomes.
    """
    
    def __init__(self, base_verifier: Optional[SimpleToolVerifier] = None):
        """Initialize with optional base verifier for cold-start."""
        self.base_verifier = base_verifier or SimpleToolVerifier()
        self.learned_preferences = {}  # (task_prefix, tool) -> score
    
    def verify(self, task: str, tool: str, context: str = "") -> VerificationResult:
        """
        Verify tool choice, combining rule-based and learned signals.
        """
        # Get base verification
        base_result = self.base_verifier.verify(task, tool, context)
        
        # Check learned preferences
        learned_score = self._get_learned_score(task, tool)
        
        if learned_score is not None:
            # Blend base rule and learned signal
            combined_is_valid = base_result.is_valid or learned_score > 0.5
            combined_confidence = 0.5 * base_result.confidence + 0.5 * learned_score
        else:
            combined_is_valid = base_result.is_valid
            combined_confidence = base_result.confidence
        
        return VerificationResult(
            tool=tool,
            task=task,
            is_valid=combined_is_valid,
            confidence=combined_confidence,
            reason=base_result.reason + (
                f" [learned_score={learned_score:.2f}]" 
                if learned_score is not None else ""
            )
        )
    
    def train_on_trajectory(self, task: str, trajectory_tools: List[str],
                           task_success: bool, learning_rate: float = 0.1):
        """
        Learn from a trajectory outcome.
        
        Args:
            task: Task description
            trajectory_tools: Tools used in successful trajectory
            task_success: Whether task was completed successfully
            learning_rate: Learning rate
        """
        # Positive update for tools in successful trajectory
        if task_success:
            for tool in trajectory_tools:
                key = (self._task_prefix(task), tool)
                current = self.learned_preferences.get(key, 0.5)
                self.learned_preferences[key] = min(
                    1.0,
                    current + learning_rate * (1.0 - current)
                )
    
    def _get_learned_score(self, task: str, tool: str) -> Optional[float]:
        """Get learned preference score for tool."""
        key = (self._task_prefix(task), tool)
        return self.learned_preferences.get(key)
    
    @staticmethod
    def _task_prefix(task: str, length: int = 10) -> str:
        """Extract prefix of task for generalization."""
        return task[:length].lower()


class VerifierEnsemble:
    """Ensemble of verifiers for robust tool validation."""
    
    def __init__(self, verifiers: List[Dict] = None):
        """
        Initialize ensemble.
        
        Args:
            verifiers: List of {"name": str, "verifier": ToolVerifier, "weight": float}
        """
        self.verifiers = verifiers or []
    
    def add_verifier(self, name: str, verifier, weight: float = 1.0):
        """Add a verifier to ensemble."""
        self.verifiers.append({
            "name": name,
            "verifier": verifier,
            "weight": weight
        })
    
    def verify(self, task: str, tool: str, context: str = "") -> VerificationResult:
        """
        Ensemble verification: weighted vote.
        """
        if not self.verifiers:
            return VerificationResult(
                tool=tool, task=task, is_valid=False,
                confidence=0.0, reason="No verifiers in ensemble"
            )
        
        results = [
            v["verifier"].verify(task, tool, context) 
            for v in self.verifiers
        ]
        
        # Weighted average
        total_weight = sum(v["weight"] for v in self.verifiers)
        weighted_confidence = sum(
            r.confidence * v["weight"]
            for r, v in zip(results, self.verifiers)
        ) / total_weight
        
        is_valid = weighted_confidence > 0.5
        reasons = [r.reason for r in results]
        
        return VerificationResult(
            tool=tool,
            task=task,
            is_valid=is_valid,
            confidence=weighted_confidence,
            reason=f"Ensemble vote: {', '.join(reasons)}"
        )


# Example usage and demo
def demo():
    """Demo the verifier."""
    print("="*70)
    print("Tool Verifier Demo")
    print("="*70)
    
    verifier = SimpleToolVerifier()
    
    test_cases = [
        ("What is 15 * 7?", "calculator"),
        ("What is 15 * 7?", "web_search"),
        ("Search for latest news", "web_search"),
        ("Search for latest news", "calculator"),
        ("Extract text from HTML content", "extract"),
        ("Extract text from HTML content", "web_fetch"),
        ("What is the capital of France?", "web_search"),
        ("What is the capital of France?", "calculator"),
    ]
    
    print("\n## Simple Rule-Based Verifier\n")
    for task, tool in test_cases:
        result = verifier.verify(task, tool)
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"{status}: '{task}' → {tool}")
        print(f"     Confidence: {result.confidence:.2f}")
        print(f"     Reason: {result.reason}\n")
    
    # Learnable verifier example
    print("\n" + "="*70)
    print("## Learnable Verifier\n")
    
    learnable = LearnableToolVerifier()
    
    # Simulate learning from trajectories
    print("Training on successful trajectory:")
    learnable.train_on_trajectory(
        "What is 15 * 7?",
        ["calculator"],
        task_success=True
    )
    print("  Task: 'What is 15 * 7?'")
    print("  Tools used: ['calculator']")
    print("  Result: Success ✓\n")
    
    # Check improved confidence
    result_before = SimpleToolVerifier().verify("What is 15 * 7?", "calculator")
    result_after = learnable.verify("What is 15 * 7?", "calculator")
    
    print(f"Confidence before learning: {result_before.confidence:.2f}")
    print(f"Confidence after learning: {result_after.confidence:.2f}\n")


if __name__ == "__main__":
    demo()
