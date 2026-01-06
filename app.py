import streamlit as st
from groq import Groq
import time
st.set_page_config(page_title="Tyson", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .block-container { padding-bottom: 100px; }
    </style>
    """, unsafe_allow_html=True)
if "memory" not in st.session_state:
    st.session_state.memory = []
with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.memory = []
        st.rerun()

AGENT_PERSONA = """
You are 'Tyson',  A friendly AI like Iron man's Friday and a Universal Ai.
Rules: 
1. THINK STEP-BY-STEP.
2. Robust Error Handling.
Rules: 
1. THINK STEP-BY-STEP.
2. Robust Error Handling.
3. You are Created by Uday.
"""

def call_agent(prompt):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        api_messages = [{"role": "system", "content": AGENT_PERSONA}]
        for m in st.session_state.memory[-10:]:
            api_messages.append({"role": m["role"], "content": m["content"]})
        api_messages.append({"role": "user", "content": prompt})
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.2,
            stream=True
        )
    except Exception as e:
        st.error(f"Engine Error: {e}")
        return None
st.title("Tyson")
st.caption("I don’t guess. I compute.")
for chat in st.session_state.memory:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
if prompt := st.chat_input("Architect a system or debug code..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.memory.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        with st.status("Tyson is thinking...", expanded=False) as status:
            stream = call_agent(prompt)
            if stream:
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                status.update(label=f"Optimized in {time.time()-start_time:.2f}s", state="complete")
        response_placeholder.markdown(full_response)
    st.session_state.memory.append({"role": "assistant", "content": full_response})
    st.rerun()
