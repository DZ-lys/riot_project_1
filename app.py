import streamlit as st
from rag_backend import LoLRAG
#import os
#from dotenv import load_dotenv

# Load environment variables
#load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LoL Knowledge Agent",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat appearance
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .strategy-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .neo4j { background-color: #4A90E2; color: white; }
    .pinecone { background-color: #7B68EE; color: white; }
    .hybrid { background-color: #FF6B6B; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag' not in st.session_state:
    # Initialize RAG system
    try:
        st.session_state.rag = LoLRAG(
            openai_api_key=st.secrets['OPENAI_API_KEY'],
            pinecone_api_key=st.secrets['PINECONE_API_KEY'],
            pinecone_index_name='lol-rag'
        )
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        st.session_state.error = str(e)

# Sidebar
with st.sidebar:
    st.title("🎮 LoL Knowledge Agent")
    st.markdown("---")
    
    # System status
    if st.session_state.get('initialized', False):
        st.success("✅ System Ready")
    else:
        st.error("❌ Initialization Failed")
        if 'error' in st.session_state:
            st.error(st.session_state.error)
    
    st.markdown("---")
    
    # Example questions
    st.subheader("💡 Example Questions")
    
    examples = [
        "Which champions have shields?",
        "Find abilities that stun",
        "Show me all mages",
        "Give me abilities with grounded effect",
        "Which champions are tanks and mages?",
        "Find ranged champions with crowd control"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state.selected_example = example
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Session Stats")
    st.metric("Questions Asked", len(st.session_state.messages) // 2)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Hybrid RAG System**
        
        - 🗄️ Neo4j: Structured queries
        - 🔍 Pinecone: Semantic search
        - 🤖 GPT-4: Answer generation
        
        **Data:**
        - 172 Champions
        - ~860 Abilities
        - Rich relationships
        """)

# Main chat interface
st.title("💬 Ask Me About League of Legends!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            
            # Strategy badge
            strategy_class = meta['strategy']
            st.markdown(
                f'<span class="strategy-badge {strategy_class}">🔍 {meta["strategy"].upper()}</span>',
                unsafe_allow_html=True
            )
            
            # Expandable details
            with st.expander("📋 Details"):
                st.write(f"**Reasoning:** {meta['reasoning']}")
                
                if 'cypher_query' in meta:
                    st.code(meta['cypher_query'], language='cypher')
                
                if 'semantic_query' in meta:
                    st.write(f"**Semantic Query:** {meta['semantic_query']}")
                
                if 'result_count' in meta:
                    st.write(f"**Results Found:** {meta['result_count']}")
                
                if message.get('sources'):
                    st.write(f"**Sources:** {', '.join(message['sources'][:5])}")

# Handle example question selection
if 'selected_example' in st.session_state:
    prompt = st.session_state.selected_example
    del st.session_state.selected_example
else:
    # Chat input
    prompt = st.chat_input("Ask about champions, abilities, roles, etc...")

# Process user input
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                # Get answer from RAG system
                result = st.session_state.rag.ask_question(prompt)
                
                # Display answer
                st.markdown(result['answer'])
                
                # Show strategy badge
                strategy_class = result['strategy']
                st.markdown(
                    f'<span class="strategy-badge {strategy_class}">🔍 {result["strategy"].upper()}</span>',
                    unsafe_allow_html=True
                )
                
                # Expandable details
                with st.expander("📋 Details"):
                    st.write(f"**Reasoning:** {result['reasoning']}")
                    
                    if 'cypher_query' in result['metadata']:
                        st.code(result['metadata']['cypher_query'], language='cypher')
                    
                    if 'semantic_query' in result['metadata']:
                        st.write(f"**Semantic Query:** {result['metadata']['semantic_query']}")
                    
                    if 'result_count' in result['metadata']:
                        st.write(f"**Results Found:** {result['metadata']['result_count']}")
                    
                    if result.get('sources'):
                        st.write(f"**Sources:** {', '.join(result['sources'][:5])}")
                
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "metadata": result['metadata'],
                    "sources": result['sources']
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.875rem;'>"
    "Powered by Neo4j + Pinecone + GPT-4 | 172 Champions | ~860 Abilities"
    "</div>",
    unsafe_allow_html=True
)
