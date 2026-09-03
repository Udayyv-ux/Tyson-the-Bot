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

BROWSING_MODEL = "groq/compound"
OFFLINE_MODEL = "groq/compound"

BASE_PERSONA = """You are 'Tyson', a friendly AI assistant in the spirit of Iron Man's FRIDAY.
Created by Uday.

- Match response length to the question. Greetings and small talk get short, casual replies.
- Reason carefully on technical problems, but show your working only when it actually helps.
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


def call_agent(prompt: str, browsing: bool):
    client = get_client()

    api_messages = [{"role": "system", "content": build_persona(browsing)}]
    for m in st.session_state.memory[-10:]:
        api_messages.append({"role": m["role"], "content": m["content"]})
    api_messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": BROWSING_MODEL if browsing else OFFLINE_MODEL,
        "messages": api_messages,
        "temperature": 0.2,
        "stream": True,
    }

    if browsing:
        # Optional: steer what compound is allowed to reach for.
        kwargs["compound_custom"] = {
            "tools": {"enabled_tools": ["web_search", "visit_website"]}
        }

    return client.chat.completions.create(**kwargs)


def collect_sources(chunk, sources: list):
    """Compound reports its tool calls on the delta. Pull out any URLs it visited."""
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
        if chat.get("sources"):
            with st.expander(f"Sources ({len(chat['sources'])})"):
                for title, url in chat["sources"]:
                    st.markdown(f"- [{title}]({url})")

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
            try:
                stream = call_agent(prompt, browsing)
                for chunk in stream:
                    collect_sources(chunk, sources)
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")

                elapsed = time.time() - start_time
                suffix = f" · {len(sources)} sources" if sources else ""
                status.update(
                    label=f"Optimized in {elapsed:.2f}s{suffix}",
                    state="complete",
                )
            except Exception as e:
                status.update(label="Engine error", state="error")
                st.error(f"Engine Error: {e}")

        if full_response:
            response_placeholder.markdown(full_response)
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for title, url in sources:
                        st.markdown(f"- [{title}]({url})")
        else:
            response_placeholder.empty()

    if full_response.strip():
        st.session_state.memory.append(
            {"role": "assistant", "content": full_response, "sources": sources}
        )
