
# Meta's LLaMA 3 instruct special tokens
BOT = "<|begin_of_text|>"
EOT = "<|end_of_text|>"
SOF = "<|start_header_id|>"
EOF = "<|end_header_id|>"
EOM = "<|eom_id|>"  # End of message, for multi-turn conversations
EOTN = "<|eot_id|>\n"  # End of turn

def format_turn(role, content):
    return f"{SOF}{role}{EOF}\n{content.strip()}"

def build_chat_prompt(system_prompt, chat_history):
    prompt = BOT + SOF + system_prompt["role"] + EOF + "\n" + system_prompt["text"] + EOTN
    # recent_turns = chat_history[-1:]  # Get messages from the last interaction only
    if len(chat_history) >= 3:
        last_user_msg = chat_history[-3]["text"]
        last_assistant_msg = chat_history[-2]["text"]
        recent_turns = [(last_user_msg, last_assistant_msg)]
        for user_msg, assistant_msg in recent_turns:
            prompt += format_turn("user", user_msg)
            prompt += EOTN
            prompt += format_turn("assistant", assistant_msg)
            prompt += EOTN
    prompt += format_turn("user", chat_history[-1]["text"])

    prompt += EOTN
    prompt += f"{SOF}assistant{EOF}"  # Assistant will continue from here
    return prompt

def build_arxiv_prompt(system_prompt, user_text):
    prompt = BOT + SOF + system_prompt["role"] + EOF + "\n" + system_prompt["text"] + EOTN
    prompt += format_turn("user", user_text)

    prompt += EOTN
    prompt += f"{SOF}assistant{EOF}"  # Assistant will continue from here
    return prompt