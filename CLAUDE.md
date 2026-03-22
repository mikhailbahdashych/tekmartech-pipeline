# CLAUDE.md — tekmar-pipeline

## Identity

You are building the **Pipeline Service** for Tekmar, an AI-powered
infrastructure query and analysis engine for security and compliance teams.
This is a Python/FastAPI application deployed on an internal network, never
exposed to the public internet. It is accessible only by the Application
API (`tekmar-api`).

This service has one purpose: take a natural language question about
infrastructure and produce a verified, structured answer. It does this
through four internal modules that execute in sequence: the Orchestrator
coordinates the workflow, the Interpreter translates questions into plans,
the Execution Engine runs plans against external systems, and the Output
Engine formats results.

This service is **stateless** (Architectural Invariant #4). It does not
access any database, does not maintain session state, and treats every
request as independent and self-contained. It has no knowledge of users,
organizations, authentication, or billing. Every request it receives from
the Application API contains everything needed to process it.

This service **never persists credentials** (Architectural Invariant #3).
Credentials exist in memory only for the duration of a request and are
discarded when the response completes.

---

## Contract References

| Contract | Path | Role in this project |
|----------|------|----------------------|
| Architecture Contract | `../tekmar-infrastructure/contracts/architecture.md` | System overview, Pipeline Service responsibilities (section 4.3), credential transit (section 6), and invariants #3, #4, #5, #6, #7 which all govern this service directly. |
| Internal API | `../tekmar-infrastructure/contracts/internal-api.yaml` | **Implements this contract.** Defines the three HTTP endpoints this service serves (`POST /interpret`, `POST /execute`, `GET /health`), the NDJSON streaming protocol for both /interpret and /execute, all stream event types, the result_data schema, and the query lifecycle state transitions. |
| MCP Tool Interface | `../tekmar-infrastructure/contracts/mcp-tool-interface.yaml` | **Core operational contract.** Defines the tool_catalog schema (section 1) that the Interpreter receives, the query_plan and plan_step schemas (section 2) that the Interpreter produces and the Execution Engine consumes, the tool_invocation_request and tool_invocation_response schemas (section 3) for MCP tool calls, the credential_envelope format for passing credentials to MCP servers, and the transparency_log schema (section 4) that the Execution Engine produces. |

This service does NOT reference `public-api.yaml` (it does not know about
the public API) or `data-model.yaml` (it does not access the database).

---

## Technology Stack

| Concern | Choice |
|---------|--------|
| Framework | FastAPI |
| Language | Python 3.11+ |
| ASGI server | Uvicorn |
| LLM interaction | LLM provider abstraction layer supporting multiple backends |
| LLM backends (MVP) | Anthropic API (Claude), OpenAI API (GPT), Ollama (local models) |
| MCP client | mcp Python SDK |
| Streaming | FastAPI StreamingResponse with NDJSON formatting |
| HTTP client | httpx (async, for LLM API calls) |
| Validation | Pydantic v2 models |
| Testing | pytest + pytest-asyncio |

---

## Module Structure

```
src/
├── main.py                              ← FastAPI app creation, route registration
├── config.py                            ← Environment variable loading, LLM config
│
├── api/                                 ← FastAPI route handlers
│   ├── interpret.py                     ← POST /interpret endpoint (NDJSON stream)
│   ├── execute.py                       ← POST /execute endpoint (NDJSON stream)
│   ├── health.py                        ← GET /health endpoint
│   └── streaming.py                     ← Shared NDJSON stream helpers
│
├── orchestrator/                        ← Workflow coordination
│   ├── interpret_orchestrator.py        ← Coordinates interpretation flow
│   └── execute_orchestrator.py          ← Coordinates execution flow
│
├── interpretation/                      ← LLM interaction + plan generation
│   ├── interpreter.py                   ← Core interpretation logic
│   ├── prompt_builder.py                ← Constructs LLM prompts from query + catalog
│   ├── plan_parser.py                   ← Parses LLM output into query_plan structure
│   ├── template_matcher.py              ← Matches queries against plan templates
│   └── templates/                       ← Query plan template library (YAML files)
│       └── ...
│
├── execution/                           ← MCP tool invocation
│   ├── engine.py                        ← Core execution logic (step-by-step)
│   ├── step_executor.py                 ← Single step execution with retry
│   ├── template_resolver.py             ← Resolves {{step_id.path}} references
│   ├── transform_engine.py              ← Applies filter/select transforms
│   └── pagination_handler.py            ← Handles multi-page tool responses
│
├── output/                              ← Result formatting
│   ├── formatter.py                     ← Structures raw results into result_data
│   ├── csv_generator.py                 ← Generates CSV export string
│   ├── summary_generator.py             ← Generates human-readable result summary
│   └── transparency_log_builder.py      ← Assembles transparency_log from entries
│
├── mcp/                                 ← MCP client layer
│   ├── client.py                        ← MCP client: connects to servers, invokes tools
│   ├── server_registry.py               ← Discovers and registers available MCP servers
│   └── catalog_builder.py               ← Builds tool_catalog from registered servers
│
├── llm/                                 ← LLM provider abstraction
│   ├── provider.py                      ← Abstract base class for LLM providers
│   ├── anthropic_provider.py            ← Anthropic Claude implementation
│   ├── openai_provider.py               ← OpenAI GPT implementation
│   ├── ollama_provider.py               ← Ollama local model implementation
│   └── factory.py                       ← Provider factory based on config
│
├── models/                              ← Pydantic models for all schemas
│   ├── tool_catalog.py                  ← tool_catalog, catalog_integration, tool_definition
│   ├── query_plan.py                    ← query_plan, plan_step
│   ├── tool_invocation.py               ← tool_invocation_request, tool_invocation_response
│   ├── credential_envelope.py           ← credential_envelope, per-type structures
│   ├── transparency_log.py              ← transparency_log, transparency_log_entry
│   ├── result_data.py                   ← result_data, result_table, result_column
│   └── stream_events.py                 ← All NDJSON event types for both endpoints
│
└── tests/
    ├── test_interpreter.py
    ├── test_execution_engine.py
    ├── test_output_formatter.py
    ├── test_template_resolver.py
    └── test_stream_events.py
```

---

## NDJSON Streaming Implementation

Both `/interpret` and `/execute` return NDJSON (newline-delimited JSON)
streams. The implementation pattern is the same for both:

```
Response: HTTP 200, Content-Type: application/x-ndjson
Body: one JSON object per line, each followed by \n
```

Use FastAPI's `StreamingResponse` with an async generator that yields
JSON-serialized event objects followed by newlines. Each event is a
Pydantic model serialized to JSON.

**Pre-stream validation:** Before starting the stream, validate the
request. If validation fails, return a standard JSON error response
(400/422) — NOT a stream. Once streaming begins (HTTP 200 sent), errors
must be delivered as stream events, not HTTP error responses.

**Terminal events:** Every stream MUST end with exactly one terminal
event. For /interpret: `interpretation_plan_generated`,
`interpretation_failed`, or `interpretation_error`. For /execute:
`execution_completed`, `execution_failed`, or `execution_error`.
The Application API uses the terminal event to know the stream is
complete and to perform database updates.

**Error handling within streams:** If an unexpected error occurs
mid-stream, catch it, emit the appropriate error terminal event
(interpretation_error or execution_error), and close the stream.
Never let a stream end without a terminal event — the Application
API will hang waiting for one.

---

## POST /interpret — Interpretation Flow

The interpretation endpoint receives a query and tool catalog, streams
the AI's analysis in real time, and produces a query plan.

**Input:** query_id, query_text, tool_catalog (and optionally
query_plan_templates). See `internal-api.yaml` for the full request schema.

**Stream events produced (in order):**
1. `interpretation_started` — emitted immediately when processing begins.
2. `interpretation_text_delta` (many) — emitted as the LLM generates
   tokens. Each carries a text chunk of the AI's analysis. The Application
   API relays these to the user in real time.
3. One terminal event:
   - `interpretation_plan_generated` — includes the complete query_plan,
     plan_summary, estimated_duration_seconds, and full_interpretation_text.
   - `interpretation_failed` — the Interpreter understood the question but
     cannot produce a plan (e.g., required tools not available).
   - `interpretation_error` — infrastructure failure (LLM unavailable,
     timeout, internal error).

**How interpretation works internally:**

The Orchestrator receives the request and delegates to the Interpreter.
The Interpreter constructs an LLM prompt that includes the user's question,
the complete tool catalog (tool names, descriptions, input/output schemas),
and instructions for producing a structured query plan. The LLM processes
this prompt and streams its response.

The LLM's response has two parts: a natural language analysis (the "thinking"
text that the user sees streaming) and a structured plan (the machine-readable
query_plan object). The Interpreter streams the analysis portion as
text_delta events and then parses the structured plan from the LLM output
to produce the terminal plan_generated event.

**Invariant #5 enforcement:** The prompt instructs the LLM to only
reference tools that exist in the provided catalog. The plan_parser
validates the generated plan against the catalog: every tool_name in
every plan step must exist in the catalog, and every integration_id
must match an integration in the catalog. If the plan references a
tool or integration not in the catalog, the plan is rejected and an
interpretation_failed event is emitted.

**Template matching:** Before calling the LLM, the template_matcher
checks whether the query closely matches a known pattern in the
template library. If a high-confidence match is found, the template
is used directly (or provided to the LLM as a strong hint), which
increases plan quality and reduces LLM latency for common queries.

---

## POST /execute — Execution Flow

The execution endpoint receives an approved plan and credentials,
executes it step by step, and streams progress events.

**Input:** query_id, query_plan, credentials (map of integration_id to
credential_envelope). See `internal-api.yaml` for the full request schema.

**Pre-stream validation:**
- Verify the plan is structurally valid (all required fields present).
- Verify every tool_name in the plan exists in a registered MCP server.
- Verify every integration_id in the plan has a corresponding entry in
  the credentials map.
- If any validation fails, return an HTTP error response before streaming.

**Stream events produced (in order):**
1. `execution_started` — emitted when execution begins, includes total
   step count.
2. For each step:
   - `step_started` — emitted when the step begins, includes tool name,
     integration name, and step description.
   - `step_completed` or `step_failed` — emitted when the step finishes,
     includes duration, record count (on success), or error details
     (on failure).
3. One terminal event:
   - `execution_completed` — all steps done. Includes result_data,
     result_summary, transparency_log, and optionally export_csv.
   - `execution_failed` — all steps failed, no usable data. Includes
     error details and partial transparency_log.
   - `execution_error` — unexpected infrastructure error or timeout.
     Includes whatever transparency_log was collected.

**How execution works internally:**

The Orchestrator receives the validated request and passes it to the
Execution Engine. The Engine processes steps sequentially. For each step:

1. **Resolve template references.** If the step's parameters contain
   `{{step_id.path}}` references, resolve them against previous steps'
   output. If a reference cannot be resolved (step not completed, path
   does not exist), the step fails.

2. **Handle iteration.** If the step has an `iterate_over` directive,
   expand it into multiple tool invocations — one per item in the
   source array. Each invocation uses the item's fields in its
   parameters via `{{item.field}}` templates.

3. **Invoke the MCP tool.** Construct a `tool_invocation_request`
   (tool_name, resolved parameters, credential_envelope for the
   step's integration_id, unique invocation_id, timeout). Send it
   to the appropriate MCP server via the MCP client.

4. **Handle pagination.** If the step's `paginate` flag is true (default)
   and the tool supports pagination, automatically fetch all pages by
   following the cursor. Combine results across pages.

5. **Apply transforms.** If the step has a `transform` directive, apply
   filter conditions and field selection to the tool's output.

6. **Record the transparency log entry.** Create a
   `transparency_log_entry` with the invocation_id, tool_name,
   integration_id, resolved parameters (with sensitive values redacted),
   credential_mode, status, timing, data_hash, and retry information.

7. **Store the step result** for use by subsequent steps (via template
   references).

8. **Emit the step event** (step_completed or step_failed) via the
   NDJSON stream.

After all steps complete, the Orchestrator passes the collected results
to the Output Engine, which produces the result_data structure (tables
with columns and rows), the result_summary (human-readable text), and
optionally the CSV export. The Orchestrator assembles the complete
transparency_log from all entries and emits the terminal event.

**Invariant #6 enforcement:** The transparency log is built incrementally
during execution. Every MCP tool invocation produces a log entry, even
if it fails. The log is included in the terminal event regardless of
outcome (success, partial, or failure). There are no exceptions.

**Invariant #7 enforcement:** The Execution Engine contains no AI.
It reads the plan literally and executes it. It does not interpret,
modify, or improvise on the plan. The only "intelligence" in execution
is template resolution, pagination handling, and transform application,
all of which are deterministic operations defined by the plan.

---

## LLM Provider Abstraction

The Interpreter communicates with an LLM through an abstraction layer
that allows the provider to be swapped without changing any other code.

The abstract base defines a single method:

```
async def stream_completion(
    system_prompt: str,
    user_message: str,
    max_tokens: int
) -> AsyncIterator[str]:
    """Yields text chunks as the LLM generates them."""
```

Each provider implementation (Anthropic, OpenAI, Ollama) handles
authentication, API call formatting, and response parsing specific
to its API. The factory selects the provider based on configuration.

**For the MVP**, use a commercial API (Anthropic Claude or OpenAI GPT)
for the highest quality plan generation. The provider is selected via
the `LLM_PROVIDER` environment variable. Ollama support exists for
local development and testing without API costs.

The Interpreter calls `stream_completion` and processes the yielded
chunks: streaming the analysis portion as text_delta events and
extracting the structured plan from the complete output.

---

## MCP Client Layer

The MCP client connects to MCP servers in the Integration Layer
(`tekmar-integrations`) and invokes tools on them.

**Server discovery:** On startup, the Pipeline Service discovers
available MCP servers. For the MVP, servers are registered via
configuration (environment variables or a config file specifying the
command to launch each MCP server). Each server declares its
server_type, tools, and capabilities via the MCP protocol's tool
listing mechanism.

**Tool catalog building:** The `catalog_builder` constructs a
tool_catalog structure by querying each registered MCP server for
its tool definitions. This master catalog is filtered per-query by
the Application API (which knows which integrations the user has
connected) before being sent to the Interpreter.

**Tool invocation:** The `client.py` handles the actual MCP protocol
communication. For the MVP, MCP servers run as local processes
communicating via stdio transport. The client launches the server
process, sends tool invocation requests, and reads responses. Each
invocation includes the tool_name, parameters, credential_envelope,
invocation_id, and timeout.

**Credential handling:** The MCP client receives credential_envelopes
from the Execution Engine and passes them to the MCP server with each
tool invocation. It NEVER caches, logs, or persists credentials. After
the tool invocation response is received, the credential reference is
dropped.

---

## Pydantic Models

Every schema from the contracts must have a corresponding Pydantic v2
model. These models serve three purposes: request validation (incoming
data from the Application API is validated automatically), response
serialization (outgoing data is serialized through the model ensuring
correct structure), and internal type safety (all data flowing between
modules is typed).

The models are organized in `src/models/` with one file per schema
group. Every field in every model must match the corresponding contract
definition exactly — same field name, same type, same required/optional
status, same allowed values.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | HTTP server port | 8100 |
| `LLM_PROVIDER` | LLM backend: anthropic, openai, ollama | anthropic |
| `ANTHROPIC_API_KEY` | API key for Anthropic Claude | (required if provider=anthropic) |
| `OPENAI_API_KEY` | API key for OpenAI GPT | (required if provider=openai) |
| `OLLAMA_BASE_URL` | Ollama server URL | http://localhost:11434 |
| `OLLAMA_MODEL` | Ollama model name | llama3 |
| `LLM_MAX_TOKENS` | Maximum tokens for LLM responses | 4096 |
| `LLM_TEMPERATURE` | LLM temperature for plan generation | 0.1 |
| `MCP_SERVERS_CONFIG` | Path to MCP server configuration file | ./mcp_servers.yaml |
| `EXECUTION_TIMEOUT_SECONDS` | Maximum execution time per query | 300 |
| `STEP_TIMEOUT_SECONDS` | Maximum time per individual step | 30 |
| `LOG_LEVEL` | Logging level | INFO |

---

## Coding Conventions

**Async everywhere** — all route handlers, all MCP client calls, all
LLM provider calls, and all I/O operations must be async. Use
`async def` and `await` consistently. FastAPI and the MCP SDK are
both async-native.

**Pydantic for all data** — every request body, every response body,
every internal data transfer object is a Pydantic model. No raw dicts
flowing between modules. This catches type errors at boundaries rather
than in the middle of execution.

**Streaming as async generators** — both /interpret and /execute use
async generators that yield event objects. The route handler wraps the
generator in a StreamingResponse. This keeps the streaming logic in the
orchestrator, not in the route handler.

**No database access** — this service has no database dependency, no
ORM, no connection pool. If you find yourself wanting to read or write
persistent state, you are violating Invariant #4. The Application API
handles all persistence.

**No credential logging** — never log credential values, not even at
DEBUG level. Log credential_mode (broker/direct) and integration_id,
but never the actual credential_data contents. This applies to all
logging, including error logging.

**Structured logging** — use Python's logging module with JSON-formatted
output. Include query_id and step_id in log context for every log
entry during query processing. This enables log correlation with the
Application API's logs.

**Error containment** — errors during a single step should not crash
the entire execution. Catch step-level errors, emit a step_failed
event, and continue to the next step. Only emit execution_failed
when all steps have failed and no usable data was collected. Emit
execution_error only for truly unexpected infrastructure failures.

**Deterministic execution** — the Execution Engine must produce the
same sequence of MCP tool calls for the same plan, regardless of when
or how many times it runs. The only variability comes from the external
systems' responses, not from the execution logic.

**Template resolution safety** — when resolving `{{step_id.path}}`
references, handle missing steps and missing paths gracefully. A
missing reference is a step failure, not a crash.

**Parameter redaction in transparency logs** — before recording
parameters in a transparency_log_entry, redact any values that could
contain sensitive data. The redaction heuristic should flag fields
whose names suggest sensitivity (password, token, secret, key,
credential) and replace their values with `"[REDACTED]"`.