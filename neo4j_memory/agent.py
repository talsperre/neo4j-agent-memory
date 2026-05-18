import asyncio
import json
import os
import re
from contextlib import asynccontextmanager

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_EFFORT = "xhigh"
EMBEDDING_MODEL = "text-embedding-3-small"

MAX_OUTPUT_TOKENS = 4096
MESSAGE_TIMEOUT_S = 240
TOOL_TIMEOUT_S = 30
MAX_TOOL_RESULT_CHARS = 12_000

CYPHER_WRITE_TOOLS = {"write_neo4j_cypher"}
CYPHER_READ_TOOLS = {"read_neo4j_cypher"}
CYPHER_TOOLS = CYPHER_WRITE_TOOLS | CYPHER_READ_TOOLS

DESTRUCTIVE_CYPHER = re.compile(r"\b(DELETE|DETACH\s+DELETE|DROP|REMOVE)\b", re.I)

EMBED_TOOL_NAME = "compute_embedding"
EMBED_TOOL = {
    "name": EMBED_TOOL_NAME,
    "description": (
        "Compute OpenAI text embeddings (text-embedding-3-small, 1536 dims). "
        "Use this BEFORE writing a Fact (pass the result as the $embedding "
        "param) and BEFORE issuing a vector query (pass as $q_embedding). "
        "Returns a list of vectors, one per input text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Up to 32 texts per call. Keep each under ~500 tokens.",
                "maxItems": 32,
            }
        },
        "required": ["texts"],
    },
}


@asynccontextmanager
async def neo4j_mcp_session():
    params = StdioServerParameters(
        command="uvx",
        args=["mcp-neo4j-cypher"],
        env={
            **os.environ,
            "NEO4J_URI": os.environ["NEO4J_URI"],
            "NEO4J_USERNAME": os.environ["NEO4J_USER"],
            "NEO4J_PASSWORD": os.environ["NEO4J_PASSWORD"],
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def mcp_tools_to_anthropic(mcp_tools) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in mcp_tools
    ]


async def run_agent_loop(
    client: anthropic.AsyncAnthropic,
    openai_client: AsyncOpenAI,
    mcp: ClientSession,
    system_prompt: str,
    user_text: str,
    *,
    required_user_id: str,
    max_iterations: int = 100,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
) -> str:
    tools_response = await mcp.list_tools()
    tools = mcp_tools_to_anthropic(tools_response.tools) + [EMBED_TOOL]

    messages: list[dict] = [
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    ]

    for _ in range(max_iterations):
        response = await _next_message(client, system_prompt, tools, messages, model, effort)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return _text_content(response.content)

        tool_results = [
            await _tool_result(
                block,
                mcp=mcp,
                openai_client=openai_client,
                required_user_id=required_user_id,
            )
            for block in response.content
            if block.type == "tool_use"
        ]
        if not tool_results:
            return _text_content(response.content)
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"agent exhausted iteration budget ({max_iterations})")


async def _next_message(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    tools: list[dict],
    messages: list[dict],
    model: str,
    effort: str,
):
    system = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    async with asyncio.timeout(MESSAGE_TIMEOUT_S):
        async with client.messages.stream(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            system=system,
            tools=tools,
            messages=messages,
        ) as stream:
            return await stream.get_final_message()


def _text_content(blocks) -> str:
    return "".join(block.text for block in blocks if block.type == "text")


async def _tool_result(
    block,
    *,
    mcp: ClientSession,
    openai_client: AsyncOpenAI,
    required_user_id: str,
) -> dict:
    try:
        content = await _dispatch_tool(
            block.name,
            block.input,
            mcp=mcp,
            openai_client=openai_client,
            required_user_id=required_user_id,
        )
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": _truncate(content),
        }
    except Exception as exc:
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": f"Error: {type(exc).__name__}: {exc}",
            "is_error": True,
        }


async def _dispatch_tool(
    name: str,
    tool_input: dict,
    *,
    mcp: ClientSession,
    openai_client: AsyncOpenAI,
    required_user_id: str,
) -> str:
    if name == EMBED_TOOL_NAME:
        return await _compute_embeddings(openai_client, tool_input["texts"])

    _validate_cypher_call(name, tool_input, required_user_id)
    async with asyncio.timeout(TOOL_TIMEOUT_S):
        result = await mcp.call_tool(name, tool_input)
    return json.dumps([c.model_dump(mode="json") for c in result.content])


async def _compute_embeddings(openai_client: AsyncOpenAI, texts: list[str]) -> str:
    async with asyncio.timeout(TOOL_TIMEOUT_S):
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
    return json.dumps([item.embedding for item in response.data])


def _validate_cypher_call(name: str, tool_input: dict, required_user_id: str) -> None:
    if name not in CYPHER_TOOLS:
        return

    query = tool_input.get("query", "")
    params = tool_input.get("params") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            raise PermissionError(
                f"{name} params must be a JSON object (got unparseable string)"
            )
    if not isinstance(params, dict):
        raise PermissionError(f"{name} params must be a JSON object")

    if name in CYPHER_WRITE_TOOLS and DESTRUCTIVE_CYPHER.search(query):
        raise PermissionError(
            "write_neo4j_cypher may not delete or drop -- supersede instead"
        )
    if params.get("user_id") != required_user_id:
        raise PermissionError(f"{name} must bind params.user_id = {required_user_id!r}")


def _truncate(content: str) -> str:
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    return content[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
