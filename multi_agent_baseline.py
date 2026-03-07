"""
Multi-Agent Baseline for Learning to Share.

Standalone implementation of a collaborative multi-agent system using local LLMs.

Key features:
- Multiple agents query the same task independently
- Simple tools: web search, fetch, extract, calculator
- Aggregates answers and tracks tool usage
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import re
import time
import os

import requests


@dataclass
class ToolResult:
    """Result from a tool call."""
    ok: bool
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    """Single step taken by an agent."""
    agent_id: int
    step_id: int
    tool_called: str
    tool_args: Dict
    tool_result: ToolResult
    response: str
    timestamp: float


@dataclass
class AgentTrajectory:
    """Full trajectory for one agent."""
    agent_id: int
    task_id: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_tool_calls: int = 0
    total_time_sec: float = 0.0


@dataclass
class MultiAgentResult:
    """Results from multi-agent run."""
    task_id: str
    individual_answers: List[str]
    aggregated_answer: Optional[str]
    total_tool_calls: int
    total_time_sec: float
    agent_trajectories: List[AgentTrajectory]


class LocalLLMAgent:
    """Single agent using local Ollama inference."""
    
    SYSTEM_PROMPT = """You are a helpful research assistant. You can use the following tools:
- web_search: Search the web for information
- web_fetch: Fetch a URL
- extract: Extract text from HTML
- calculator: Perform calculations

When using a tool, output ONLY:
```tool
{{"tool": "tool_name", "args": {{"arg1": "value1", ...}}}}
```

When ready to answer, output:
ANSWER: <your answer>

Be concise and focus on the task."""
    
    def __init__(self, agent_id: int, llm_base_url: str = "http://localhost:11434", model: str = "qwen3:8b", tavily_api_key: Optional[str] = None):
        self.agent_id = agent_id
        self.llm_base_url = llm_base_url
        self.model = model
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.trajectory: Optional[AgentTrajectory] = None
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running."""
        try:
            r = requests.get(f"{self.llm_base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.llm_base_url}. "
                f"Make sure Ollama is running: ollama serve\n{e}"
            )
    
    def solve(self, task_id: str, task: str, max_steps: int = 5) -> AgentTrajectory:
        """Solve a task using the local LLM."""
        self.trajectory = AgentTrajectory(agent_id=self.agent_id, task_id=task_id)
        t0 = time.time()
        
        notes = []
        
        for step_id in range(max_steps):
            # Build prompt with history
            history_str = "\n".join([
                f"Step {i+1}: {s.tool_called} -> {s.tool_result.content[:100]}..."
                for i, s in enumerate(self.trajectory.steps[-3:])
            ])
            
            prompt = f"""{self.SYSTEM_PROMPT}

## Task
{task}

## History
{history_str if history_str else "No steps yet"}

## Current Notes
{chr(10).join(notes[-5:]) if notes else "None"}

What's your next action?"""
            
            # Query LLM
            response = self._query_llm(prompt)
            
            # Parse tool call or answer
            if response.startswith("ANSWER:"):
                answer = response[len("ANSWER:"):].strip()
                self.trajectory.final_answer = answer
                notes.append(f"Step {step_id}: Found answer: {answer[:50]}...")
                break
            
            # Try to extract tool call
            tool_match = re.search(r'```tool\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not tool_match:
                notes.append(f"Step {step_id}: Parse error, retrying...")
                continue
            
            try:
                tool_spec = json.loads(tool_match.group(1))
                tool_name = tool_spec.get("tool", "")
                tool_args = tool_spec.get("args", {})
            except json.JSONDecodeError:
                notes.append(f"Step {step_id}: JSON parse error")
                continue
            
            # Execute tool
            result = self._execute_tool(tool_name, tool_args)
            self.trajectory.total_tool_calls += 1
            
            step = AgentStep(
                agent_id=self.agent_id,
                step_id=step_id,
                tool_called=tool_name,
                tool_args=tool_args,
                tool_result=result,
                response=response[:200],
                timestamp=time.time(),
            )
            self.trajectory.steps.append(step)
            notes.append(f"Step {step_id}: {tool_name} ok={result.ok} -> {result.content[:50]}...")
            
            if not result.ok:
                notes.append(f"  Tool failed: {result.content}")
        
        # Fallback: if no answer produced after max steps, ask LLM directly
        if self.trajectory.final_answer is None:
            history_str = "\n".join([
                f"Step {i+1}: {s.tool_called} -> {s.tool_result.content[:100]}..."
                for i, s in enumerate(self.trajectory.steps)
            ])
            prompt = f"""/nothink You are a helpful research assistant answering a question.
Do NOT use any tools. Do NOT output any code blocks. Just give a short, direct answer.

## Task
{task}

## Research Results
{history_str if history_str else "No prior research"}

Based on the information above, what is the answer? Reply with ONLY the answer, nothing else.
ANSWER:"""
            response = self._query_llm(prompt)
            if "ANSWER:" in response:
                self.trajectory.final_answer = response.split("ANSWER:")[-1].strip()
            else:
                self.trajectory.final_answer = response.strip()
        
        self.trajectory.total_time_sec = time.time() - t0
        return self.trajectory
    
    def solve_with_policy(self, task_id: str, task: str, tool_policy,
                          max_steps: int = 5) -> AgentTrajectory:
        """
        Solve a task using an external tool selection policy (for on-policy PPO).
        
        The policy network selects which tool to use at each step,
        while the LLM generates tool arguments and final answers.
        
        Args:
            task_id: Task identifier
            task: Task description/question
            tool_policy: Policy object with select_tool(context) method
            max_steps: Maximum number of steps
        
        Returns:
            AgentTrajectory with the steps taken
        """
        self.trajectory = AgentTrajectory(agent_id=self.agent_id, task_id=task_id)
        t0 = time.time()
        notes = []
        
        for step_id in range(max_steps):
            # Build context for policy
            history_str = "\n".join([
                f"Step {i+1}: {s.tool_called} -> {s.tool_result.content[:100]}..."
                for i, s in enumerate(self.trajectory.steps[-3:])
            ])
            
            context = f"Task: {task}\nHistory: {history_str if history_str else 'No steps yet'}"
            
            # Policy selects the tool
            tool_name, action_idx, log_prob, value, features = tool_policy.select_tool(context)
            
            if tool_name == "no_tool":
                # Ask LLM for direct answer (no tool instructions to avoid confusion)
                prompt = f"""/nothink You are a helpful research assistant answering a question.
Do NOT use any tools. Do NOT output any code blocks. Just give a short, direct answer.

## Task
{task}

## Research Results
{history_str if history_str else "No prior research"}

## Notes
{chr(10).join(notes[-5:]) if notes else "None"}

Based on the information above, what is the answer? Reply with ONLY the answer, nothing else.
ANSWER:"""
                response = self._query_llm(prompt)
                if "ANSWER:" in response:
                    answer = response.split("ANSWER:")[-1].strip()
                else:
                    answer = response.strip()
                self.trajectory.final_answer = answer

                # Record no_tool as a step so PPO can learn from this decision
                step = AgentStep(
                    agent_id=self.agent_id,
                    step_id=step_id,
                    tool_called="no_tool",
                    tool_args={},
                    tool_result=ToolResult(ok=True, content=answer[:200], meta={}),
                    response=f"Policy selected: no_tool",
                    timestamp=time.time(),
                )
                self.trajectory.steps.append(step)
                break
            
            # Ask LLM for tool arguments
            tool_args = self._get_tool_args(tool_name, task, history_str, notes)
            
            # Execute tool
            result = self._execute_tool(tool_name, tool_args)
            self.trajectory.total_tool_calls += 1
            
            step = AgentStep(
                agent_id=self.agent_id,
                step_id=step_id,
                tool_called=tool_name,
                tool_args=tool_args,
                tool_result=result,
                response=f"Policy selected: {tool_name}",
                timestamp=time.time(),
            )
            self.trajectory.steps.append(step)
            notes.append(f"Step {step_id}: {tool_name} ok={result.ok} -> {result.content[:50]}...")
        
        # If no answer was produced, ask LLM for final answer
        if self.trajectory.final_answer is None:
            history_str = "\n".join([
                f"Step {i+1}: {s.tool_called} -> {s.tool_result.content[:100]}..."
                for i, s in enumerate(self.trajectory.steps)
            ])
            prompt = f"""/nothink You are a helpful research assistant answering a question.
Do NOT use any tools. Do NOT output any code blocks. Just give a short, direct answer.

## Task
{task}

## Research Results
{history_str}

Based on the research above, what is the answer? Reply with ONLY the answer, nothing else.
ANSWER:"""
            response = self._query_llm(prompt)
            if "ANSWER:" in response:
                self.trajectory.final_answer = response.split("ANSWER:")[-1].strip()
            else:
                self.trajectory.final_answer = response.strip()
        
        self.trajectory.total_time_sec = time.time() - t0
        return self.trajectory
    
    def _get_tool_args(self, tool_name: str, task: str, history_str: str, notes: list) -> Dict:
        """Generate tool arguments using the LLM."""
        prompt = f"""{self.SYSTEM_PROMPT}

## Task
{task}

## History
{history_str if history_str else "No steps yet"}

## Notes
{chr(10).join(notes[-5:]) if notes else "None"}

You must use the tool: {tool_name}
Generate ONLY the tool call with arguments:
```tool
{{"tool": "{tool_name}", "args": {{"""
        
        response = self._query_llm(prompt)
        
        # Try to parse tool args from response
        tool_match = re.search(r'\{[^{}]*"args"\s*:\s*(\{[^{}]*\})', response, re.DOTALL)
        if tool_match:
            try:
                return json.loads(tool_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try simpler JSON extraction
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if "args" in parsed:
                    return parsed["args"]
                return parsed
            except json.JSONDecodeError:
                pass
        
        # Fallback: generate default args based on tool type
        if tool_name == "web_search":
            return {"query": task}
        elif tool_name == "calculator":
            nums = re.findall(r'\d+', task)
            return {"expr": " + ".join(nums) if nums else "0"}
        elif tool_name == "web_fetch":
            return {"url": ""}
        elif tool_name == "extract":
            return {"html": ""}
        return {}
    
    def _query_llm(self, prompt: str, temperature: float = 0.2, max_tokens: int = 256) -> str:
        """Query Ollama for a response."""
        try:
            url = f"{self.llm_base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": "/nothink " + prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False,
            }
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except Exception as e:
            return f"LLM_ERROR: {e}"
    
    def _execute_tool(self, tool_name: str, args: Dict) -> ToolResult:
        """Execute a tool."""
        try:
            if tool_name == "web_search":
                return self._tool_search(args.get("query", ""), args.get("num", 3))
            elif tool_name == "web_fetch":
                return self._tool_fetch(args.get("url", ""))
            elif tool_name == "extract":
                return self._tool_extract(args.get("html", ""))
            elif tool_name == "calculator":
                return self._tool_calc(args.get("expr", ""))
            else:
                return ToolResult(ok=False, content=f"UNKNOWN_TOOL: {tool_name}", meta={})
        except Exception as e:
            return ToolResult(ok=False, content=f"TOOL_ERROR: {e}", meta={})
    
    def _tool_search(self, query: str, num: int = 3) -> ToolResult:
        """Real web search using Tavily API."""
        if not self.tavily_api_key:
            return ToolResult(ok=False, content="SEARCH_ERROR: TAVILY_API_KEY not set", meta={})
        
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "max_results": min(num, 10),
                "include_answer": True,
            }
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()
            
            results = data.get("results", []) or []
            content = "\n".join([
                f"- {r.get('title', 'N/A')} ({r.get('url', 'N/A')})\n  {r.get('content', '')[:100]}"
                for r in results[:num]
            ])
            
            return ToolResult(ok=True, content=content, meta={"results": results[:num]})
        except Exception as e:
            return ToolResult(ok=False, content=f"SEARCH_ERROR: {e}", meta={})
    
    def _tool_fetch(self, url: str) -> ToolResult:
        """Mock web fetch - returns dummy content."""
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            return ToolResult(ok=True, content=r.text[:1000], meta={"url": url})
        except Exception as e:
            return ToolResult(ok=False, content=f"FETCH_ERROR: {e}", meta={"url": url})
    
    def _tool_extract(self, html: str) -> ToolResult:
        """Extract text from HTML."""
        if not html:
            return ToolResult(ok=False, content="EXTRACT_ERROR: Empty HTML", meta={})
        # Simple extraction: remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
        return ToolResult(ok=True, content=text[:500], meta={})
    
    def _tool_calc(self, expr: str) -> ToolResult:
        """Safe calculator."""
        allowed = set("0123456789+-*/(). %")
        if any(ch not in allowed for ch in expr):
            return ToolResult(ok=False, content="CALC_ERROR: illegal characters", meta={})
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            if isinstance(result, (int, float)):
                return ToolResult(ok=True, content=str(result), meta={})
            return ToolResult(ok=False, content="CALC_ERROR: non-numeric result", meta={})
        except Exception as e:
            return ToolResult(ok=False, content=f"CALC_ERROR: {e}", meta={})


class MultiAgentBaseline:
    """Multi-agent baseline using local LLMs."""
    
    def __init__(self, num_agents: int = 3, llm_base_url: str = "http://localhost:11434", model: str = "qwen3:8b", tavily_api_key: Optional[str] = None):
        self.num_agents = num_agents
        self.llm_base_url = llm_base_url
        self.model = model
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.agents = [
            LocalLLMAgent(i, llm_base_url=llm_base_url, model=model, tavily_api_key=self.tavily_api_key)
            for i in range(num_agents)
        ]
    
    def solve_task(self, task_id: str, task: str, max_steps_per_agent: int = 5) -> MultiAgentResult:
        """Run multiple agents on the same task and aggregate results."""
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task[:60]}...")
        print(f"{'='*60}")
        
        trajectories = []
        answers = []
        total_tool_calls = 0
        t0 = time.time()
        
        # Run each agent
        for agent in self.agents:
            print(f"\nAgent {agent.agent_id}:")
            traj = agent.solve(task_id, task, max_steps=max_steps_per_agent)
            trajectories.append(traj)
            total_tool_calls += traj.total_tool_calls
            
            if traj.final_answer:
                answers.append(traj.final_answer)
                print(f"  Answer: {traj.final_answer[:60]}...")
            else:
                print(f"  No answer (max steps reached)")
            
            print(f"  Tool calls: {traj.total_tool_calls}, Time: {traj.total_time_sec:.1f}s")
        
        # Aggregate answers
        aggregated = self._aggregate_answers(answers)
        total_time = time.time() - t0
        
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
        """Aggregate multiple answers (simple voting or first non-empty)."""
        if not answers:
            return None
        
        # Count duplicates and return most common
        from collections import Counter
        counts = Counter(answers)
        most_common, _ = counts.most_common(1)[0]
        return most_common


def demo():
    """Demo: run multi-agent baseline."""
    baseline = MultiAgentBaseline(num_agents=2, model="qwen3:8b")
    
    tasks = [
        ("q1", "What is the capital of France?"),
        ("q2", "What is 15 * 7?"),
        ("q3", "Who won the 2024 Olympics?"),
    ]
    
    results = []
    for task_id, task in tasks:
        result = baseline.solve_task(task_id, task, max_steps_per_agent=3)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<20} {'Aggregated Answer':<40} {'Tool Calls':<12}")
    print("-" * 72)
    for r in results:
        print(f"{r.task_id:<20} {r.aggregated_answer[:38]:<40} {r.total_tool_calls:<12}")


if __name__ == "__main__":
    demo()
