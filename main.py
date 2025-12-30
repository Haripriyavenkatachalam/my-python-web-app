# ===============================
# main.py ✅ APP + API ROBUST INTEGRATION (HYBRID MODE, NO 'SORRY')
# ===============================

import builtins
import api   # your numeric API logic
import gradio as gr
import app   # your meaning-based hostel chatbot

# --------------------------------------------------
# 0. HELPER FUNCTIONS: GREETING + MEANINGFUL ANSWER FILTER
# --------------------------------------------------
def is_greeting(text):
    """
    Detects greetings like hi, hello, hey, etc.
    """
    text = str(text).strip().lower()
    greetings = [
        "hi", "hello", "hey", "hai",
        "good morning", "good afternoon", "good evening"
    ]
    return text in greetings

def is_meaningful_answer(answer, user_query=None):
    """
    Returns True if answer is meaningful:
    - Not empty or None
    - Not nonsense
    - Optionally, contains words from user_query (for App relevance)
    """
    if answer is None:
        return False

    answer_str = str(answer).strip()
    if not answer_str:
        return False

    # Ignore very short generic answers
    if len(answer_str) <= 3:
        return False

    # If user_query provided, check for shared words
    if user_query:
        query_words = [w for w in str(user_query).lower().split() if len(w) > 2]
        # If user query has no meaningful words, treat answer as meaningless
        if not query_words:
            return False
        if not any(word in answer_str.lower() for word in query_words):
            return False

    return True

# --------------------------------------------------
# 1. LOAD API SEMANTIC ENGINE
# --------------------------------------------------
API_URL = "https://api.hostelconfig.impreserp.co.in/hostelapi/api/HostelBasic/DashboardSummary"
TOKEN = "PASTE_YOUR_TRAIN_TOKEN_HERE"

print("🔄 Loading API semantic engine...")
api_data = api.get_hostel_data(API_URL, TOKEN)
df_api = api.generate_qa_data(api_data)
model_api, df_api = api.prepare_semantic_model(df_api)
print("✅ API engine ready")

# --------------------------------------------------
# 2. DEFINE api_answer
# --------------------------------------------------
def api_answer(question):
    if is_greeting(question):
        return None

    # Ask API semantic engine
    result = api.answer_semantic_question(
        question,
        df_api,
        model_api,
        threshold=0.60  # ⬅ slightly stricter
    )

    if result is None:
        return None

    answer_str = str(result).strip().lower()

    # ❌ Block weak / generic answers
    if any(x in answer_str for x in [
        "sorry",
        "couldn't understand",
        "not understand",
        "no data",
        "not available"
    ]):
        return None

    # --------------------------------------------------
    # 🔒 HARD RELEVANCE FILTER (KEY FIX)
    # --------------------------------------------------

    # meaningful words from query
    query_words = [w for w in question.lower().split() if len(w) > 2]

    if not query_words:
        return None

    # Require at least ONE strong overlap
    overlap = sum(1 for w in query_words if w in answer_str)

    if overlap == 0:
        return None

    return str(result)


# --------------------------------------------------
# 3. HELPER: CHECK IF APP ANSWER IS VALID
# --------------------------------------------------
def is_valid_app_answer(answer):
    if answer is None:
        return False
    answer_str = str(answer).strip().lower()
    if not answer_str:
        return False
    invalid_phrases = [
        "sorry",
        "couldn't understand",
        "could not understand",
        "not understand"
    ]
    return not any(phrase in answer_str for phrase in invalid_phrases)

# --------------------------------------------------
# 4. UNIFIED CHATBOT LOGIC: HYBRID (BOTH APP + API)
# --------------------------------------------------
from api import preprocess_text

def unified_chatbot(user_query):
    if is_greeting(user_query):
        return "👋 Hello! How can I help you with hostel information?"

    # ✅ Normalize ONCE
    normalized_query = preprocess_text(user_query)

    user_query_words = [w for w in normalized_query.split() if len(w) > 2]

    # 1️⃣ Check API first (USE normalized query)
    api_ans = api_answer(normalized_query)
    if not is_meaningful_answer(api_ans, normalized_query):
        api_ans = None

    # 2️⃣ Get APP answer (USE normalized query)
    app_ans, app_link, app_img = app.hostel_chatbot(normalized_query)

    app_ans = str(app_ans) if app_ans else ""
    app_link = str(app_link) if app_link else ""
    app_img = str(app_img) if app_img else ""

    # ❌ Ignore invalid App answers
    if (
        not is_valid_app_answer(app_ans)
        or not is_meaningful_answer(app_ans, normalized_query)
        or not user_query_words
    ):
        app_ans = ""
        app_link = ""
        app_img = ""

    # 3️⃣ Compose final response
    response_parts = []

    if api_ans:
        response_parts.append(api_ans)

    if app_ans:
        response_parts.append(app_ans)
        if app_link:
            response_parts.append(f"🔗 {app_link}")
        if app_img:
            response_parts.append(f"🖼️ {app_img}")

    # 🚫 Nothing found
    if not response_parts:
        return "❌ Sorry, I couldn't find an answer. Please ask more clearly or provide more details."

    return "\n\n".join(response_parts)


# --------------------------------------------------
# 5. GRADIO CHAT FUNCTION
# --------------------------------------------------
def chat(user_message, history):
    history = history or []
    if not user_message.strip():
        return history, ""
    bot_text = unified_chatbot(user_message)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_text})
    return history, ""

# --------------------------------------------------
# 6. LAUNCH GRADIO UI
# --------------------------------------------------
with gr.Blocks(title="Unified Hostel Chatbot") as demo:
    gr.Markdown("## 🏨 Hostel Chatbot")
    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Ask hostel related questions...", show_label=False)
    with gr.Row():
        send = gr.Button("Send")
        clear = gr.Button("Clear")
    send.click(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear.click(lambda: [], outputs=chatbot)

demo.launch(share=True)

