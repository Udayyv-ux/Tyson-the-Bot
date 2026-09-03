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

# Models: compound is Groq's agentic system (web search + page visits built in).
BROWSING_MODEL = "groq/compound"
# compound-mini runs one tool call per turn instead of iterating, so it can't
# balloon its own context. Used as the retry when compound returns a 413.
FALLBACK_MODEL = "groq/compound-mini"
OFFLINE_MODEL = "llama-3.3-70b-versatile"

# Rough character budget for conversation history. Compound's answers are long,
# so trimming by message count alone will eventually blow past the request limit.
MAX_CONTEXT_CHARS = 12000

# Keep compound's own tool context small. Raise if answers feel too thin.
MAX_SEARCH_RESULTS = 3

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
- Lead with the most recent information and name your sources inline.
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
    """Walk backwards from the newest message, keeping whatever fits in budget."""
    kept, used = [], 0
    for m in reversed(st.session_state.memory):
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
        "stream": True,
    }

    if browsing:
        # web_search only. visit_website pulls whole page bodies into compound's
        # own context and is the usual cause of a server-side 413 on this model.
        kwargs["compound_custom"] = {
            "tools": {
                "enabled_tools": ["web_search"],
                "search_settings": {"max_results": MAX_SEARCH_RESULTS},
            }
        }

    return client.chat.completions.create(**kwargs)


def collect_sources(chunk, sources: list) -> None:
    """Compound reports tool calls on the delta. Pull out any URLs it visited."""
    delta = chunk.choices[0].delta
    tools = getattr(delta, "executed_tools", None)
    if not tools:
        return
    for tool in tools:
        output = getattr(tool, "output", None) or {}
        if isinstance(output, dict):
            for result in output.get("results", []) or []:
                url = result.get("url")
                title = result.get("title") or url
                if url and url not in [s[1] for s in sources]:
                    sources.append((title, url))


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for title, url in sources:
            st.markdown(f"- [{title}]({url})")


def is_too_large(err: Exception) -> bool:
    text = str(err)
    return "Entity Too Large" in text or "413" in text


# ---------------- UI ----------------

with st.sidebar:
    st.subheader("Controls")
    browsing = st.toggle(
        "Live web browsing",
        value=True,
        help="Lets Tyson search the web and read pages for current information.",
    )
    st.caption(f"Engine: `{BROWSING_MODEL if browsing else OFFLINE_MODEL}`")
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
        response_placeholder = st.empty()
        full_response = ""
        sources = []
        start_time = time.time()

        label = "Tyson is searching the web..." if browsing else "Tyson is thinking..."
        with st.status(label, expanded=False) as status:

            def consume(stream):
                """Drain a stream into full_response / sources."""
                global full_response
                for chunk in stream:
                    collect_sources(chunk, sources)
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")

            try:
                consume(call_agent(prompt, browsing))
                elapsed = time.time() - start_time
                suffix = f" · {len(sources)} sources" if sources else ""
                status.update(
                    label=f"Optimized in {elapsed:.2f}s{suffix}",
                    state="complete",
                )
            except Exception as e:
                if is_too_large(e):
                    # Request overflowed. Retry on the lighter agent with no
                    # history: compound-mini makes a single tool call per turn,
                    # so its context can't snowball the way compound's does.
                    status.update(
                        label="Too large — retrying on lighter engine",
                        state="running",
                    )
                    full_response = ""
                    sources = []
                    try:
                        consume(
                            call_agent(
                                prompt,
                                browsing,
                                history=False,
                                model=FALLBACK_MODEL,
                            )
                        )
                        status.update(
                            label=f"Answered via {FALLBACK_MODEL}", state="complete"
                        )
                    except Exception as e2:
                        status.update(label="Engine error", state="error")
                        st.error(f"Engine Error: {e2}")
                else:
                    status.update(label="Engine error", state="error")
                    st.error(f"Engine Error: {e}")

        if full_response:
            response_placeholder.markdown(full_response)
            render_sources(sources)
        else:
            response_placeholder.empty()

    if full_response.strip():
        st.session_state.memory.append(
            {"role": "assistant", "content": full_response, "sources": sources}
        )
