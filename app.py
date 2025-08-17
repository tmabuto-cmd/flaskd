import os
import uuid
from flask import Flask, request, jsonify, render_template, make_response
from transformers import AutoModelForCausalLM, AutoTokenizer,BlenderbotForConditionalGeneration,BlenderbotTokenizerFast
import torch

app = Flask(__name__)

# Choose a small, chat-tuned causal LM.
# You can swap this string for another HF model id (e.g., "microsoft/DialoGPT-large")
MODEL_NAME = os.environ.get("CHAT_MODEL", "facebook/blenderbot-400M-distill")

print(f"Loading model: {MODEL_NAME} (this may take a moment)...")
tokenizer = BlenderbotTokenizerFast.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()

# Simple in-memory history store: { session_id: [("user","..."),("bot","..."), ...] }
# For production, replace with Redis or a DB.
CONV_HISTORY = {}

# Helper: format history into a single prompt for causal LM
def build_prompt(history, user_msg, max_turns=6):
    """
    DialoGPT was trained on Reddit conversations using special tokens. A simple way
    to condition on history is concatenating prior turns. We'll keep the last `max_turns`
    turns to avoid very long inputs.
    """
    turns = history[-max_turns:] if history else []
    prompt_text = ""
    for role, text in turns:
        if role == "user":
            prompt_text += f"User: {text} "
        else:
            prompt_text += f"Bot: {text} "
    prompt_text += f"User: {user_msg} Bot:" 
    return prompt_text

# Helper: generate a reply
@torch.inference_mode()
def generate_reply(prompt_text, max_new_tokens=200, temperature=0.7, top_p=0.9):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    # Truncate if input is too long for the model
    max_ctx = 1024  # DialoGPT-medium context window
    if input_ids.shape[1] > max_ctx:
        input_ids = input_ids[:, -max_ctx:]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Decode only the newly generated part
    gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    # Post-process a bit: stop at first hard newline if it rambles
    gen_text = gen_text.strip()
    # Sometimes DialoGPT continues with "User:" — trim that if it appears
    if "User:" in gen_text:
        gen_text = gen_text.split("User:")[0].strip()
    return gen_text

@app.route("/")
def index():
    # Assign a session id via cookie if not present
    session_id = request.cookies.get("chat_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    resp = make_response(render_template("index.html"))
    resp.set_cookie("chat_session_id", session_id, max_age=60*60*24*7)  # 7 days
    return resp

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Get session id from cookie or body
    session_id = request.cookies.get("chat_session_id") or data.get("session_id") or str(uuid.uuid4())
    hist = CONV_HISTORY.get(session_id, [])

    prompt_text = build_prompt(hist, user_msg)
    bot_reply = generate_reply(prompt_text)

    # Update history (keep it from growing unbounded)
    hist.append(("user", user_msg))
    hist.append(("bot", bot_reply))
    if len(hist) > 20:
        hist = hist[-20:]
    CONV_HISTORY[session_id] = hist

    return jsonify({
        "reply": bot_reply,
        "session_id": session_id
    })

if __name__ == "__main__":
    # Run: python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
