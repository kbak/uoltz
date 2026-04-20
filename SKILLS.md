# Skills

Skills are auto-discovered from `app/skills/`. Each skill is a folder with a `skill.yaml` manifest and Python tool modules. Use `/skills` in chat to see what's loaded.

## Built-in Skills

Enabled out of the box:

| Skill | Tools | Description |
|-------|-------|-------------|
| `research` | `research_topic` | Multi-source web + news search with date verification and AI synthesis. Best for factual lookups: weather, stock prices, current events |
| `brainstorm` | `brainstorm_topic` | Multi-agent brainstorming with domain-aware specialists, fact-checking, YouTube video analysis, prior brainstorm awareness, and confidence-scored reports. Saves results to `data/brainstorms/` |
| `youtube_summary` | `summarize_youtube` | Extract YouTube transcripts and produce detailed video summaries. Handles long videos via chunked summarization |
| `summarize` | `summarize_content` | Fetch and summarize any URL or raw text. Extracts article content with BeautifulSoup, converts to markdown, produces TLDR + key points |
| `memory` | `remember`, `search_memory` | Long-term memory across conversations — save durable user preferences or stable facts; recall past context when the current window lacks needed information |

Shipped but **disabled by default** (flip `enabled: true` in the skill's `skill.yaml` to use):

| Skill | Tools | Why disabled |
|-------|-------|-------------|
| `web_search` | `web_search` | Superseded by external MCP-based search (e.g. SearXNG via the stack's mcp-proxy) in multi-service deployments |
| `notes` | `save_note`, `list_notes`, `read_note` | Niche JSON-backed note store; most users don't need it |
| `rss_digest` | `rss_digest` | Requires a FreshRSS instance; watcher services handle news in multi-service deployments |
| `shell` | `run_shell_command` | Runs arbitrary commands inside the bot container with only a trivial `startswith` blocklist — enable only in trusted environments |
| `skill_builder` | `create_skill`, `list_skills_on_disk` | Runtime LLM-generated skill scaffolding; useful for experimentation, risky to leave enabled |

## Brainstorm Skill (v2)

The brainstorm skill uses a multi-agent Graph pipeline with 8 agents:

```
decomposer ──┬── specialist*   ──┐
             ├── critic        ──┤
             ├── researcher    ──├── fact_checker ──► synthesizer
             ├── pragmatist    ──┤
             └── media_scout   ──┘

* specialist = technical_architect | strategist | creative_director | visionary
  (selected automatically based on topic domain)
```

Key capabilities:

- **Domain-aware specialists** — the decomposer classifies the topic (tech, business, creative, general) and swaps in the appropriate specialist agent
- **Fact-checking** — a dedicated fact-checker agent cross-references claims from all specialists, spot-checks via web search, and marks claims as VERIFIED/UNVERIFIED/DISPUTED
- **YouTube video analysis** — the media scout searches for relevant YouTube videos and extracts transcript insights
- **RSS feed context** — pulls recent articles from FreshRSS feeds into the decomposer's context (if configured)
- **Prior brainstorm awareness** — scans `data/brainstorms/` for past sessions on related topics to avoid rehashing
- **Confidence scoring** — the synthesizer tags each recommendation as HIGH/MEDIUM/LOW confidence based on source verification
- **User context** — accepts an optional `context` parameter (e.g. "solo developer, $5k budget") to tailor advice
- **Temporal grounding** — all agents know today's date to avoid outdated reasoning

Usage: `/brainstorm <topic>` or ask the agent to brainstorm naturally.

## Creating a New Skill

### From the template

1. Copy the template:
   ```bash
   cp -r app/skills/_template app/skills/my_skill
   ```

2. Edit `app/skills/my_skill/skill.yaml`:
   ```yaml
   name: my_skill
   description: What this skill does.
   version: "1.0.0"
   enabled: true
   tools:
     - "my_module:my_tool_function"
   ```

3. Implement your tools in `app/skills/my_skill/my_module.py` using the `@tool` decorator from Strands

4. Restart the bot — skills are auto-discovered from the `app/skills/` directory

Set `enabled: false` in `skill.yaml` to disable a skill without deleting it.

### From natural language (skill_builder, opt-in)

If you enable the `skill_builder` skill (it ships disabled), you can ask the bot to create a skill at runtime:

> "Create a skill that fetches the top Hacker News stories and summarizes them"

`skill_builder` will generate the `skill.yaml` and Python module, write them to `data/custom_skills/`, and the new skill will be available after a restart. Because the generated module runs inside the bot with no sandbox, keep this skill disabled unless you trust both the LLM and the prompts that reach it.

### Skill anatomy

```
app/skills/my_skill/
├── skill.yaml          # Manifest: name, description, tool references
├── __init__.py         # Empty (required for Python imports)
└── my_module.py        # Tool implementations with @tool decorator
```

### skill.yaml reference

```yaml
name: my_skill                          # Must be a valid Python identifier
description: What this skill does.      # Shown in /skills and system prompt
version: "1.0.0"
enabled: true                           # Set false to disable without deleting

tools:                                  # List of "module:function" references
  - "my_module:my_tool_function"

# Optional: register a direct slash command (bypasses LLM routing)
command: /mycommand
command_arg: input_param                # Parameter name to pass user input to
command_usage: "/mycommand <input>"     # Usage hint shown on empty invocation
```

### Tool implementation

```python
from strands import tool

@tool
def my_tool_function(query: str) -> str:
    """Short description of what this tool does.

    Longer description that helps the LLM decide WHEN to use this tool.
    Be specific about the use cases.

    Args:
        query: Description of the parameter.
    """
    # Your implementation here
    return "result"
```

Key conventions:
- The docstring is critical — the LLM reads it to decide when to invoke the tool
- Always return a string
- Use type hints on all parameters
- For config access: `import config` (provides `config.make_model()`, `config.llm`, etc.)
- For web search: `from skills.web_search.search import web_search`
- For formatting: append `config.formatting_instruction()` to sub-agent system prompts

### Discovery locations

Skills are loaded from two directories:

| Location | Purpose |
|----------|---------|
| `app/skills/` | Built-in skills (shipped with the bot) |
| `data/custom_skills/` | External skills (generated by skill_builder, survives container rebuilds) |
