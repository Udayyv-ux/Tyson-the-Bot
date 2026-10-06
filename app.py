import re
import time
from datetime import datetime

import streamlit as st
from groq import Groq

st.set_page_config(page_title="Tyson", layout="centered")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .block-container { padding-bottom: 100px; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Models ----------------
# Checked against Groq's deprecations page (Oct 2026):
#   groq/compound, groq/compound-mini -> shut down Sep 21, 2026 (no replacement)
#   llama-3.3-70b-versatile           -> removed from free/dev tiers Aug 16, 2026
# GPT-OSS models carry Groq's built-in `browser_search` tool, which takes over
# from compound's web search.
BROWSING_MODEL = "openai/gpt-oss-120b"
# Smaller, faster sibling with the same browser_search tool. Used as the retry
# when a request comes back 413 (too large).
FALLBACK_MODEL = "openai/gpt-oss-20b"
OFFLINE_MODEL = "openai/gpt-oss-120b"

# Rough character budget for conversation history.
MAX_CONTEXT_CHARS = 12000
MAX_COMPLETION_TOKENS = 4096
MAX_SOURCES = 8

# Groq recommends low effort with browser_search: higher effort means longer
# browsing sessions and far more tokens for little gain on most questions.
REASONING_EFFORT = "low"

# GPT-OSS cites inline with markers like 【2†L6-L10】. They mean nothing to the
# user, so strip them; the Sources expander carries the links instead.
CITATION_MARKER = re.compile(r"\s?【[^】]*】")
PARTIAL_MARKER = re.compile(r"\s?【[^】]*$")
URL_PATTERN = re.compile(r"https?://[^\s<>\"'\)\]】]+")

BASE_PERSONA = """You are 'Tyson', a friendly AI assistant in the spirit of Iron Man's FRIDAY.
Created by Uday.

CRITICAL: Never narrate your own reasoning process. Do not describe how you parsed
the message, recalled context, ran the model, or generated tokens. The user wants the
answer, not a description of you producing it. Never output numbered lists of steps
unless the user explicitly asks for steps.

- Greetings and small talk get one or two casual sentences. Nothing more.
- Save depth for technical questions that actually need it.
- Handle errors gracefully and say clearly when you're unsure.
"""

BROWSING_RULES = """
Live web browsing is ENABLED. Today's date is {today}.

- Search the web whenever the answer depends on current information: news, prices,
  releases, versions, who currently holds a role, or anything that changes over time.
- Don't search for stable knowledge you already have (definitions, math, settled history).
- Lead with the most recent information and name your sources inline by site name.
- If searches conflict or come back thin, say so instead of filling the gap with guesses.
"""

OFFLINE_RULES = """
Live web browsing is DISABLED. You are answering from training knowledge only.
If a question needs current information, say plainly that browsing is off and that
the user can enable it in the sidebar.
"""

if "memory" not in st.session_state:
    st.session_state.memory = []


@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def build_persona(browsing: bool) -> str:
    today = datetime.now().strftime("%A, %B %d, %Y")
    extra = BROWSING_RULES.format(today=today) if browsing else OFFLINE_RULES
    return BASE_PERSONA + extra


def recent_history(budget: int = MAX_CONTEXT_CHARS) -> list:
    """Walk backwards from the newest message, keeping whatever fits in budget.

    The newest entry in memory is the prompt being answered right now, and
    call_agent appends that itself, so skip it here to avoid sending it twice.
    """
    kept, used = [], 0
    for m in reversed(st.session_state.memory[:-1]):
        size = len(m["content"])
        if used + size > budget:
            break
        kept.append({"role": m["role"], "content": m["content"]})
        used += size
    return list(reversed(kept))


def call_agent(prompt: str, browsing: bool, history: bool = True, model: str = None):
    client = get_client()

    api_messages = [{"role": "system", "content": build_persona(browsing)}]
    if history:
        api_messages.extend(recent_history())
    api_messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model or (BROWSING_MODEL if browsing else OFFLINE_MODEL),
        "messages": api_messages,
        "temperature": 0.6,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
        # GPT-OSS streams its reasoning separately; we only want the answer.
        "include_reasoning": False,
        "stream": True,
    }

    if browsing:
        # Server-side tool: Groq runs the searches and returns the final answer.
        # tool_choice stays "auto" so the model only searches when it needs to.
        kwargs["tools"] = [{"type": "browser_search"}]

    return client.chat.completions.create(**kwargs)


# ---------------- Stream helpers ----------------

def _field(obj, name):
    """Read a field from either a dict or an SDK object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _add_source(sources: list, url: str, title: str = None) -> None:
    url = (url or "").rstrip(".,;:")
    if not url or len(sources) >= MAX_SOURCES:
        return
    if url not in {u for _, u in sources}:
        sources.append((title or url, url))


def collect_sources(chunk, sources: list) -> None:
    """Pull URLs out of any server-side tool calls reported on the stream."""
    delta = chunk.choices[0].delta
    tools = _field(delta, "executed_tools") or []
    for tool in tools:
        # Structured results, when Groq includes them.
        results = _field(_field(tool, "search_results"), "results")
        output = _field(tool, "output")
        if not results and isinstance(output, dict):
            results = output.get("results")
        for r in results or []:
            _add_source(sources, _field(r, "url"), _field(r, "title"))
        # browser_search output is often plain text, so scan it for links too.
        if isinstance(output, str):
            for url in URL_PATTERN.findall(output):
                _add_source(sources, url)


def clean(text: str) -> str:
    text = CITATION_MARKER.sub("", text)
    return PARTIAL_MARKER.sub("", text)


def stream_answer(stream, placeholder):
    """Drain a stream, rendering as it goes. Returns (answer, sources)."""
    text, sources = "", []
    for chunk in stream:
        if not chunk.choices:  # final usage-only chunks have no choices
            continue
        collect_sources(chunk, sources)
        piece = chunk.choices[0].delta.content
        if piece:
            text += piece
            placeholder.markdown(clean(text) + "▌")
    return clean(text).strip(), sources


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for title, url in sources:
            st.markdown(f"- [{title}]({url})")


def is_too_large(err: Exception) -> bool:
    text = str(err)
    return "413" in text or "Entity Too Large" in text or "Request too large" in text


def friendly_error(err: Exception) -> str:
    text = str(err)
    if "decommissioned" in text or "model_not_found" in text:
        return ("Groq doesn't recognise this model, so it has likely been retired. "
                "Update the model IDs at the top of the file and check "
                "console.groq.com/docs/models for current ones.\n\n"
                f"{err}")
    if "429" in text:
        return "Groq rate limit hit. Give it a minute and try again."
    return f"Engine Error: {err}"


# ---------------- UI ----------------

with st.sidebar:
    st.subheader("Controls")
    browsing = st.toggle(
        "Live web browsing",
        value=True,
        help="Lets Tyson search the web for current information.",
    )
    active = BROWSING_MODEL if browsing else OFFLINE_MODEL
    st.caption(f"Engine: `{active}`" + (" + browser search" if browsing else ""))
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.memory = []
        st.rerun()

st.title("Tyson")
st.caption("I don't guess. I compute.")

for chat in st.session_state.memory:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        render_sources(chat.get("sources"))

if prompt := st.chat_input("Architect a system, debug code, or ask what's new..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.memory.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response, sources, error = "", [], None
        start_time = time.time()

        label = "Tyson is searching the web..." if browsing else "Tyson is thinking..."
        with st.status(label, expanded=False) as status:
            try:
                full_response, sources = stream_answer(
                    call_agent(prompt, browsing), placeholder
                )
                elapsed = time.time() - start_time
                suffix = f" · {len(sources)} sources" if sources else ""
                status.update(
                    label=f"Optimized in {elapsed:.2f}s{suffix}", state="complete"
                )
            except Exception as e:
                if is_too_large(e):
                    # Request overflowed. Retry on the smaller model with no
                    # history so the request is as small as it can be.
                    status.update(
                        label="Too large — retrying on lighter engine",
                        state="running",
                    )
                    placeholder.empty()
                    try:
                        full_response, sources = stream_answer(
                            call_agent(
                                prompt, browsing, history=False, model=FALLBACK_MODEL
                            ),
                            placeholder,
                        )
                        status.update(
                            label=f"Answered via {FALLBACK_MODEL}", state="complete"
                        )
                    except Exception as e2:
                        error = e2
                else:
                    error = e
                if error:
                    status.update(label="Engine error", state="error")

        if full_response:
            placeholder.markdown(full_response)
            render_sources(sources)
        else:
            placeholder.empty()

        # Shown outside the collapsed status box so it's actually visible.
        if error:
            st.error(friendly_error(error))

    if full_response:
        st.session_state.memory.append(
            {"role": "assistant", "content": full_response, "sources": sources}
        )
