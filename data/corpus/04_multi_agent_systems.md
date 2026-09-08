# Multi-Agent AI Systems: Architecture and Failure Modes

## What Are Multi-Agent AI Systems?

A multi-agent AI system (MAS) consists of multiple autonomous AI agents that collaborate,
communicate, and coordinate to accomplish complex tasks that would be difficult for a
single agent. Common architectures include:
- **Pipeline agents**: Agents run in sequence (Researcher → Critic → Writer)
- **Parallel agents**: Agents run simultaneously and merge results
- **Hierarchical agents**: A coordinator agent delegates to specialist sub-agents

## Retrieval-Augmented Generation (RAG)

RAG systems combine a language model with external knowledge retrieval. Instead of
relying solely on the model's training data, the agent retrieves relevant documents
from a corpus before generating a response. This reduces hallucination and allows
the system to work with up-to-date or domain-specific information.

## Common Failure Modes

According to the MAST study (NeurIPS 2025), which analyzed 1,600+ failure traces
from multi-agent systems, 14 distinct failure types were identified. The most common include:

1. **Semantic thrashing**: An agent repeatedly asks slightly different versions of the
   same question, unable to make progress on a task the corpus cannot answer.
2. **Context overflow**: The agent's context window is exhausted by accumulated messages.
3. **Hallucination cascade**: One agent hallucinates a fact, and downstream agents build
   on that incorrect information.
4. **Tool failure**: An external tool (e.g., search API) fails and the agent cannot
   handle the failure gracefully.

## The Recursion Limit Problem

LangGraph and similar frameworks impose a `recursion_limit` to prevent infinite loops.
When this limit is hit, the framework raises an exception (e.g., `GraphRecursionError`)
and terminates the entire run. This blunt approach discards all progress made and
provides no useful partial output to the caller.

The 2026 MAESTRO study found that 75.17% of multi-agent failures never trigger a
system exception — they silently degrade in output quality without the system
detecting anything is wrong.
