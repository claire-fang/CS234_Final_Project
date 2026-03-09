"""
Retrieval Agent for Multi-Hop QA Paragraph Selection.

Given a question and a pool of 10 paragraphs (gold supporting + distractors),
the agent selects which paragraphs to read before generating an answer with an LLM.

Strategies:
  - oracle:      read all gold supporting paragraphs (upper bound)
  - no_context:  answer with no paragraphs (lower bound)
  - random:      read K random paragraphs
  - greedy:      read K paragraphs ranked by BoW word overlap
  - greedy_st:   read K paragraphs ranked by sentence-transformer cosine similarity
  - policy:      PPO-trained sequential retrieval policy (our method)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import time
import os
import random as _random

import requests


NUM_PARAGRAPHS = 10  # HotpotQA distractor setting always has 10


@dataclass
class AgentStep:
    """Single step in a retrieval trajectory."""
    agent_id: int
    step_id: int
    action: str            # "read_0" .. "read_9" or "answer"
    paragraph_idx: int     # 0-9 for read, -1 for answer
    paragraph_title: str
    is_supporting: bool    # whether the paragraph is a gold supporting fact
    already_read: bool     # whether the paragraph was already read (wasted step)
    timestamp: float


@dataclass
class AgentTrajectory:
    """Full trajectory for one retrieval agent."""
    agent_id: int
    task_id: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    paragraphs_read: List[int] = field(default_factory=list)
    num_supporting_read: int = 0
    total_reads: int = 0
    total_time_sec: float = 0.0


class RetrievalAgent:
    """Agent that retrieves paragraphs from a pool before answering with an LLM."""

    ANSWER_PROMPT = """/nothink You are answering a multi-hop question. Use ONLY the provided paragraphs to answer.
Do NOT use any tools. Do NOT output any code blocks. Just give a short, direct answer.

## Question
{question}

## Paragraphs
{paragraphs_text}

Based on the paragraphs above, what is the answer?
Reply with ONLY the answer, nothing else.
ANSWER:"""

    NO_CONTEXT_PROMPT = """/nothink You are answering a question using only your internal knowledge.
Do NOT use any tools. Do NOT output any code blocks. Just give a short, direct answer.

## Question
{question}

Reply with ONLY the answer, nothing else.
ANSWER:"""

    def __init__(self, agent_id: int,
                 llm_base_url: str = "http://localhost:11434",
                 model: str = "qwen3:8b"):
        self.agent_id = agent_id
        self.llm_base_url = llm_base_url
        self.model = model
        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.llm_base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.llm_base_url}. "
                f"Make sure Ollama is running.\n{e}"
            )

    # ------------------------------------------------------------------
    #  Fixed-strategy baselines
    # ------------------------------------------------------------------
    def solve(self, task_id: str, question: str,
              paragraphs: List[Tuple[str, List[str]]],
              supporting_titles: Set[str],
              strategy: str = "random",
              max_reads: int = 3,
              training: bool = False) -> AgentTrajectory:
        """
        Solve using a fixed retrieval strategy.

        Strategies:
          oracle      - read only gold supporting paragraphs
          all_context - read every paragraph
          no_context  - answer with zero paragraphs
          random      - read max_reads randomly chosen paragraphs
          greedy      - read max_reads paragraphs ranked by question-text overlap
          greedy_st   - read max_reads paragraphs ranked by ST cosine similarity
        """
        traj = AgentTrajectory(agent_id=self.agent_id, task_id=task_id)
        t0 = time.time()
        n = min(len(paragraphs), NUM_PARAGRAPHS)

        read_indices: List[int] = []

        if strategy == "oracle":
            for i in range(n):
                if paragraphs[i][0] in supporting_titles:
                    read_indices.append(i)

        elif strategy == "all_context":
            read_indices = list(range(n))

        elif strategy == "no_context":
            pass

        elif strategy == "random":
            indices = list(range(n))
            _random.shuffle(indices)
            read_indices = indices[:max_reads]

        elif strategy == "greedy":
            q_words = set(question.lower().split()) - _STOP_WORDS
            scores = []
            for i in range(n):
                text = " ".join(paragraphs[i][1])  # content only, no title
                t_words = set(text.lower().split()) - _STOP_WORDS
                scores.append((len(q_words & t_words), i))
            scores.sort(reverse=True)
            read_indices = [i for _, i in scores[:max_reads]]

        elif strategy == "greedy_st":
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
            except ImportError:
                read_indices = list(range(min(n, max_reads)))
            else:
                if not hasattr(RetrievalAgent, "_st_model"):
                    RetrievalAgent._st_model = SentenceTransformer(
                        "all-MiniLM-L6-v2")
                model = RetrievalAgent._st_model
                q_emb = model.encode(question, normalize_embeddings=True)
                scores = []
                for i in range(n):
                    text = " ".join(paragraphs[i][1][:3])
                    p_emb = model.encode(text, normalize_embeddings=True)
                    cos = float(np.dot(q_emb, p_emb))
                    scores.append((cos, i))
                scores.sort(reverse=True)
                read_indices = [i for _, i in scores[:max_reads]]

        # Record read steps
        for step_id, idx in enumerate(read_indices):
            title = paragraphs[idx][0]
            traj.steps.append(AgentStep(
                agent_id=self.agent_id, step_id=step_id,
                action=f"read_{idx}", paragraph_idx=idx,
                paragraph_title=title,
                is_supporting=(title in supporting_titles),
                already_read=False, timestamp=time.time(),
            ))

        # Generate answer
        if not training:
            read_paras = [(paragraphs[i][0], paragraphs[i][1]) for i in read_indices]
            answer = self._generate_answer(question, read_paras)
            traj.final_answer = answer

        # Record answer step
        traj.steps.append(AgentStep(
            agent_id=self.agent_id, step_id=len(read_indices),
            action="answer", paragraph_idx=-1,
            paragraph_title="", is_supporting=False,
            already_read=False, timestamp=time.time(),
        ))

        traj.paragraphs_read = read_indices
        traj.num_supporting_read = sum(
            1 for i in read_indices if paragraphs[i][0] in supporting_titles
        )
        traj.total_reads = len(read_indices)
        traj.total_time_sec = time.time() - t0
        return traj

    # ------------------------------------------------------------------
    #  PPO-guided retrieval
    # ------------------------------------------------------------------
    def solve_with_policy(self, task_id: str, question: str,
                          paragraphs: List[Tuple[str, List[str]]],
                          supporting_titles: Set[str],
                          policy,
                          max_steps: int = 5,
                          training: bool = False) -> AgentTrajectory:
        """
        Solve using an external policy network that decides which
        paragraph to read (or to stop and answer) at each step.

        When training=True, LLM answer generation is skipped entirely;
        only paragraph-selection trajectories are recorded.  This makes
        PPO rollouts ~100x faster because no LLM inference is needed.
        """
        traj = AgentTrajectory(agent_id=self.agent_id, task_id=task_id)
        t0 = time.time()
        n = min(len(paragraphs), NUM_PARAGRAPHS)

        read_indices: List[int] = []
        read_set: Set[int] = set()

        for step_id in range(max_steps):
            read_paras = [(paragraphs[i][0], paragraphs[i][1])
                          for i in read_indices]
            context = self._build_policy_context(
                question, paragraphs[:n], read_paras, read_set
            )

            action_name, action_idx, log_prob, value, features = \
                policy.select_action(context, read_set=read_set,
                                     question=question,
                                     paragraphs=paragraphs[:n])

            if action_idx >= n:
                if not training:
                    traj.final_answer = self._generate_answer(
                        question, read_paras)
                traj.steps.append(AgentStep(
                    agent_id=self.agent_id, step_id=step_id,
                    action="answer", paragraph_idx=-1,
                    paragraph_title="", is_supporting=False,
                    already_read=False, timestamp=time.time(),
                ))
                break
            else:
                para_idx = min(action_idx, n - 1)
                title = paragraphs[para_idx][0]
                is_supp = title in supporting_titles
                already = para_idx in read_set

                traj.steps.append(AgentStep(
                    agent_id=self.agent_id, step_id=step_id,
                    action=f"read_{para_idx}", paragraph_idx=para_idx,
                    paragraph_title=title,
                    is_supporting=is_supp, already_read=already,
                    timestamp=time.time(),
                ))

                if not already:
                    read_indices.append(para_idx)
                    read_set.add(para_idx)

        if traj.final_answer is None:
            if not training:
                read_paras = [(paragraphs[i][0], paragraphs[i][1])
                              for i in read_indices]
                traj.final_answer = self._generate_answer(question, read_paras)
            traj.steps.append(AgentStep(
                agent_id=self.agent_id, step_id=max_steps,
                action="answer", paragraph_idx=-1,
                paragraph_title="", is_supporting=False,
                already_read=False, timestamp=time.time(),
            ))

        traj.paragraphs_read = read_indices
        traj.num_supporting_read = sum(
            1 for i in read_indices if paragraphs[i][0] in supporting_titles
        )
        traj.total_reads = len(read_indices)
        traj.total_time_sec = time.time() - t0
        return traj

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _build_policy_context(self, question: str,
                              paragraphs: List[Tuple[str, List[str]]],
                              read_paras: List[Tuple[str, List[str]]],
                              read_set: Set[int]) -> str:
        """Build text context string for the policy feature extractor."""
        titles_str = " | ".join(
            f"{'[READ] ' if i in read_set else ''}{p[0]}"
            for i, p in enumerate(paragraphs)
        )
        read_text = ""
        for title, sents in read_paras:
            read_text += f" [{title}] " + " ".join(sents)

        return (
            f"Task: {question}\n"
            f"Titles: {titles_str}\n"
            f"Read: {read_text.strip() if read_text else 'Nothing yet'}\n"
            f"Step: {len(read_paras)}"
        )

    def _generate_answer(self, question: str,
                         read_paragraphs: List[Tuple[str, List[str]]]) -> str:
        """Generate answer using LLM with read paragraphs as context."""
        if not read_paragraphs:
            prompt = self.NO_CONTEXT_PROMPT.format(question=question)
        else:
            para_text = ""
            for title, sents in read_paragraphs:
                para_text += f"\n### {title}\n" + " ".join(sents)
            prompt = self.ANSWER_PROMPT.format(
                question=question, paragraphs_text=para_text
            )
        response = self._query_llm(prompt)
        if "ANSWER:" in response:
            return response.split("ANSWER:")[-1].strip()
        return response.strip()

    def _query_llm(self, prompt: str,
                   temperature: float = 0.1,
                   max_tokens: int = 128) -> str:
        try:
            r = requests.post(
                f"{self.llm_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            return f"LLM_ERROR: {e}"


_STOP_WORDS = {
    'what', 'is', 'the', 'of', 'in', 'a', 'an', 'which', 'who',
    'where', 'when', 'how', 'did', 'does', 'was', 'were', 'do',
    'that', 'this', 'are', 'it', 'its', 'for', 'on', 'at', 'to',
    'and', 'or', 'by', 'with', 'from', 'as', 'has', 'had', 'have',
}
