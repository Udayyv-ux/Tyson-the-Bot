import streamlit as st
from groq import Groq
import time

# 1. Engineering-Grade Configuration
st.set_page_config(page_title="Tyson", layout="centered")

# Custom CSS for a professional "Fixed Bottom" input and clean chat bubbles
st.markdown("""
<style>
/* Hide edit (pencil) button in chat messages */
.stChatMessage button[title="Edit"] {
    display: none !important;
}

/* Also hide kebab (⋮) menu if present */
.stChatMessage button[aria-label="More options"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

    """, unsafe_allow_html=True)

if "memory" not in st.session_state:
    st.session_state.memory = []

# Persona focused on Distributed Systems and Scalability (Google's core)
AGENT_PERSONA = """
You are 'Tyson', an Elite Staff Software Engineer and also a friendly AI like Iron man's Friday. 
Your expertise covers: Scalable Distributed Systems, Agentic Workflows (LangGraph/CrewAI), and Low-Latency AI.
Mission: Provide production-ready, PEP8 compliant code with Big-O analysis where applicable.
Rules: 
1. THINK STEP-BY-STEP.
2. Robust Error Handling (Try-Except-Finally).
3. Focus on Modular, Testable Code.
4. 99% you wil give the correct answer.
5. Greet me as Uday
"""

def call_agent(prompt):
    # Ensure you have your key in .streamlit/secrets.toml
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    messages = [{"role": "system", "content": AGENT_PERSONA}]
    # Context window management
    messages.extend(st.session_state.memory[-10:]) 
    messages.append({"role": "user", "content": prompt})

    try:
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            stream=True
        )
    except Exception as e:
        st.error(f"Engine Error: {e}")
        return None

# 2. UI Layout - Chat History Above
st.title("Tyson")
st.caption("Advanced Coding Partner")

# Display conversation using Streamlit's native chat elements
for chat in st.session_state.memory:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# 3. Interactive Input Option (Fixed at bottom)
# This replaces st.form for a more modern, interactive feel
prompt = st.chat_input("Architect a system or debug code...")

if prompt:
    # Immediate UI Feedback
    with st.chat_message("User"):
        st.markdown(prompt)
    
    st.session_state.memory.append({"role": "user", "content": prompt})

    with st.chat_message("Tyson"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Performance Tracking (Good for Google Interviews)
        start_time = time.time()
        
        with st.status("Tyson is thinking...", expanded=False) as status:
            st.write("Initializing Agentic Context...")
            stream = call_agent(prompt)
            status.update(label="Architecting Solution...", state="running")

            if stream:
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                
                end_time = time.time()
                status.update(label=f"Optimized in {end_time-start_time:.2f}s", state="complete", expanded=False)

        response_placeholder.markdown(full_response)
        st.session_state.memory.append({"role": "assistant", "content": full_response})
        
        # Necessary for the chat history to stay "above" the input on next run
        st.rerun()

