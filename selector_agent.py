"""
Agent with Trained Tool Selector Verification.

Similar to MultiAgentBaseline, but with the ability to verify tool selections
using a trained PPO selector model at EACH STEP. The agent and selector 
collaborate interactively:
1. Agent decides on a tool (or answer)
2. Selector provides feedback/verification
3. If match -> execute tool
4. If mismatch -> add feedback to context, agent rethinks
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import os
import torch
import json
import re

from multi_agent_baseline import (
    MultiAgentBaseline, LocalLLMAgent, MultiAgentResult, 
    AgentTrajectory, AgentStep, ToolResult
)
from ppo_finetuner import PPOFineTuner, TaskScorer
from tool_verifier import SimpleToolVerifier


class InteractiveSelectorAgent(LocalLLMAgent):
    """
    Agent that uses trained selector for step-by-step verification.
    
    At each step:
    1. Agent proposes a tool or answer
    2. Selector verifies the proposal
    3. If selector agrees -> execute tool
    4. If selector disagrees -> add feedback to context, agent rethinks
    """
    
    def __init__(self, agent_id: int, 
                 llm_base_url: str = "http://localhost:11434", 
                 model: str = "qwen3:8b", 
                 tavily_api_key: Optional[str] = None,
                 selector_model: Optional[PPOFineTuner] = None,
                 verifier: Optional[SimpleToolVerifier] = None):
        """
        Initialize interactive selector agent.
        
        Args:
            agent_id: Agent identifier
            llm_base_url: Ollama server URL
            model: LLM model name
            tavily_api_key: API key for web search
            selector_model: Trained PPOFineTuner for verification
            verifier: SimpleToolVerifier for tool validity checking
        """
        super().__init__(agent_id, llm_base_url, model, tavily_api_key)
        self.selector_model = selector_model
        self.verifier = verifier or SimpleToolVerifier()
        self.selector_feedback_history = []
    
    def solve(self, task_id: str, task: str, max_steps: int = 5, 
              available_tools: List[str] = None) -> AgentTrajectory:
        """
        Solve task with step-by-step selector verification.
        
        At each step, selector provides feedback on agent's tool choice.
        """
        self.available_tools = available_tools
        self.trajectory = AgentTrajectory(agent_id=self.agent_id, task_id=task_id)
        self.selector_feedback_history = []
        t0 = __import__('time').time()
        
        notes = []
        
        for step_id in range(max_steps):
            # Build prompt with history and selector feedback
            history_str = "\n".join([
                f"Step {i+1}: {s.tool_called} -> {s.tool_result.content[:100]}..."
                for i, s in enumerate(self.trajectory.steps[-3:])
            ])
            
            # Add selector feedback to context
            feedback_str = "\n".join(self.selector_feedback_history[-3:]) if self.selector_feedback_history else "None"
            
            prompt = f"""{self.SYSTEM_PROMPT}

## Task
{task}

## History
{history_str if history_str else "No steps yet"}

## Selector Feedback
{feedback_str}

## Current Notes
{chr(10).join(notes[-5:]) if notes else "None"}

What's your next action?"""
            
            # Step 1: Agent proposes action
            response = self._query_llm(prompt)
            
            # Parse agent's proposal
            if response.startswith("ANSWER:"):
                answer = response[len("ANSWER:"):].strip()
                self.trajectory.final_answer = answer
                notes.append(f"Step {step_id}: Agent proposed answer: {answer[:50]}...")
                
                # # Get selector feedback on answer
                # if self.selector_model:
                #     selector_feedback = self._get_selector_feedback_for_answer(answer, task)
                #     notes.append(f"  Selector feedback: {selector_feedback}")
                #     self.selector_feedback_history.append(f"Step {step_id}: {selector_feedback}")
                
                break
            
            # Try to extract tool call
            tool_match = re.search(r'```tool\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not tool_match:
                notes.append(f"Step {step_id}: Parse error, retrying...")
                self.selector_feedback_history.append(
                    f"Step {step_id}: Could not parse your tool call, please try again with proper format"
                )
                continue
            
            try:
                tool_spec = json.loads(tool_match.group(1))
                tool_name = tool_spec.get("tool", "")
                tool_args = tool_spec.get("args", {})
            except json.JSONDecodeError:
                notes.append(f"Step {step_id}: JSON parse error")
                self.selector_feedback_history.append(
                    f"Step {step_id}: Invalid JSON format, please check your tool specification"
                )
                continue
            
            # Step 2: Get selector verification
            selector_decision, selector_feedback = self._verify_tool_with_selector(tool_name, task)
            print(f"  Agent proposed tool: {tool_name}, Selector decision: {selector_decision}, Feedback: {selector_feedback}")
            notes.append(f"Step {step_id}: Agent proposed {tool_name}, Selector: {selector_decision}")
            
            if not selector_decision:
                # Step 4: Selector disagrees - add feedback and agent rethinks
                notes.append(f"  Mismatch! Adding feedback: {selector_feedback}")
                self.selector_feedback_history.append(f"Step {step_id}: {selector_feedback}")
                continue
            
            # Step 3: Selector agrees - execute tool
            result = self._execute_tool(tool_name, tool_args)
            self.trajectory.total_tool_calls += 1
            
            step = AgentStep(
                agent_id=self.agent_id,
                step_id=step_id,
                tool_called=tool_name,
                tool_args=tool_args,
                tool_result=result,
                response=response[:200],
                timestamp=__import__('time').time(),
            )
            self.trajectory.steps.append(step)
            notes.append(f"  Executed: {tool_name} -> {result.content[:50]}...")
            
            # Add execution confirmation to feedback
            if result.ok:
                self.selector_feedback_history.append(
                    f"Step {step_id}: Tool {tool_name} executed successfully"
                )
            else:
                self.selector_feedback_history.append(
                    f"Step {step_id}: Tool {tool_name} failed: {result.content[:50]}"
                )
            
            if not result.ok:
                notes.append(f"  Tool failed: {result.content}")
        
        self.trajectory.total_time_sec = __import__('time').time() - t0
        return self.trajectory
    
    def _verify_tool_with_selector(self, tool_name: str, task: str) -> tuple:
        """
        Step 2: Query selector to verify if tool is appropriate for task.
        
        Returns:
            (decision: bool, feedback: str)
        """
        if not self.selector_model:
            # No selector, approve tool
            return True, f"Tool {tool_name} is acceptable"
        
        try:
            # Get selector's tool recommendations
            recommended_tools = self.selector_model.predict_tools(task, num_tools=2)
            
            # Check if agent's tool matches selector's recommendations
            if tool_name in recommended_tools:
                feedback = f"Tool {tool_name} matches selector recommendation (recommended: {recommended_tools})"
                return True, feedback
            else:
                feedback = f"Tool {tool_name} not recommended. Selector suggests: {recommended_tools}. Please reconsider."
                return False, feedback
        
        except Exception as e:
            return True, f"Could not verify with selector: {e}"
    
    def _get_selector_feedback_for_answer(self, answer: str, task: str) -> str:
        """Get selector feedback on proposed answer."""
        try:
            # Check answer validity using verifier as proxy
            # In a real system, could have more sophisticated answer evaluation
            return f"Answer recorded: '{answer[:50]}...'"
        except Exception as e:
            return f"Could not verify answer: {e}"


class MultiAgentWithSelector(MultiAgentBaseline):
    """Multi-agent system using interactive selector agents."""
    
    def __init__(self, num_agents: int = 1, 
                 llm_base_url: str = "http://localhost:11434", 
                 model: str = "qwen3:8b", 
                 tavily_api_key: Optional[str] = None,
                 selector_model_path: Optional[str] = None):
        """
        Initialize multi-agent system with selector verification.
        
        Args:
            num_agents: Number of agents
            llm_base_url: Ollama server URL
            model: LLM model name
            tavily_api_key: API key for web search
            selector_model_path: Path to trained selector model
        """
        self.num_agents = num_agents
        self.llm_base_url = llm_base_url
        self.model = model
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        # Load selector model
        self.selector_model = None
        if selector_model_path and os.path.exists(selector_model_path):
            self.selector_model = self._load_selector(selector_model_path)
        
        self.verifier = SimpleToolVerifier()
        
        # Create interactive selector agents
        self.agents = [
            InteractiveSelectorAgent(
                i, 
                llm_base_url=llm_base_url, 
                model=model, 
                tavily_api_key=self.tavily_api_key,
                selector_model=self.selector_model,
                verifier=self.verifier
            )
            for i in range(num_agents)
        ]
    
    def _load_selector(self, model_path: str) -> Optional[PPOFineTuner]:
        """Load trained selector model."""
        try:
            scorer = TaskScorer()
            selector = PPOFineTuner(scorer, device="cpu")
            selector.load_model(model_path)
            print(f"✓ Loaded selector from {model_path}")
            return selector
        except Exception as e:
            print(f"Warning: Could not load selector: {e}")
            return None
    
    def solve_task(self, task_id: str, task: str, max_steps_per_agent: int = 5, 
                   available_tools: List[str] = None) -> MultiAgentResult:
        """
        Run agents with step-by-step selector verification.
        
        Each agent interactively uses selector feedback at each step.
        """
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task[:60]}...")
        if self.selector_model:
            print(f"Using trained selector for step-by-step verification")
        if available_tools:
            print(f"Available tools: {available_tools}")
        print(f"{'='*60}")
        
        trajectories = []
        answers = []
        total_tool_calls = 0
        t0 = __import__('time').time()
        
        # Run each agent
        for agent in self.agents:
            print(f"\nAgent {agent.agent_id}:")
            traj = agent.solve(task_id, task, max_steps=max_steps_per_agent, 
                             available_tools=available_tools)
            trajectories.append(traj)
            total_tool_calls += traj.total_tool_calls
            
            if traj.final_answer:
                answers.append(traj.final_answer)
                print(f"  Answer: {traj.final_answer[:60]}...")
            else:
                print(f"  No answer (max steps reached)")
            
            print(f"  Tool calls: {traj.total_tool_calls}, Time: {traj.total_time_sec:.1f}s")
            
            # Show selector feedback history
            if agent.selector_feedback_history:
                print(f"  Selector feedback interventions: {len(agent.selector_feedback_history)}")
        
        # Aggregate answers
        aggregated = self._aggregate_answers(answers)
        total_time = __import__('time').time() - t0
        
        result = MultiAgentResult(
            task_id=task_id,
            individual_answers=answers,
            aggregated_answer=aggregated,
            total_tool_calls=total_tool_calls,
            total_time_sec=total_time,
            agent_trajectories=trajectories,
        )
        
        print(f"\nAggregated answer: {aggregated}")
        print(f"Total tool calls: {total_tool_calls}, Total time: {total_time:.1f}s")
        
        return result
    
    def _aggregate_answers(self, answers: List[str]) -> Optional[str]:
        """Aggregate multiple answers."""
        if not answers:
            return None
        from collections import Counter
        counts = Counter(answers)
        most_common, _ = counts.most_common(1)[0]
        return most_common


if __name__ == "__main__":
    import sys
    
    print("Interactive Selector Agent module loaded.")
    print("\nUsage:")
    print("  # Create multi-agent system with step-by-step selector verification")
    print("  from selector_agent import MultiAgentWithSelector")
    print("  ")
    print("  agent = MultiAgentWithSelector(")
    print("    num_agents=1,")
    print("    selector_model_path='results/hotpot_tool_selector.pt'")
    print("  )")
    print("  ")
    print("  # Solve task with selector feedback at each step")
    print("  result = agent.solve_task(")
    print("    'q1', 'What is the capital of France?'")
    print("  )")
    print("\n  Pipeline at each step:")
    print("  1. Agent proposes tool (or answer)")
    print("  2. Selector verifies if tool matches recommendations")
    print("  3. If match -> execute tool immediately")
    print("  4. If mismatch -> add selector feedback to context")
    print("  5. Agent rethinks with feedback and tries again")

