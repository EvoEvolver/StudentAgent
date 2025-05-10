class Renderer:
    def __init__():
        pass

    def render_chat_html_old(self, messages=None):
        from IPython.display import HTML
        import json
        import html
        if messages is None:
            messages = self.chat.messages

        html_parts = ["<div style='font-family:Arial, sans-serif; line-height:1.6;'>"]

        for message in messages:
            try:
                parsed = json.dumps(message)
                parsed = json.loads(parsed)["content"]['text']
            except (KeyError, json.JSONDecodeError):
                continue
            
            
            role = message.get("role", "Unknown").capitalize()
            tool_call_note = ""
            '''
            if "tool_call_id" in message:
                tool_id = message["tool_call_id"]
                tool_call_note = f" <span style='color:#666; font-size:0.85em;'>(Tool Call ID: <code>{tool_id}</code>)</span>"
            '''
            bg_color = "#e0f7fa" if role == "Assistant" else "#f8f9fa"
            text_color = "#111"
            border = "1px solid #ccc"
            border_radius = "10px"
            padding = "10px"
            margin = "10px 0"

            inner_parts = []

            if type(parsed) == str and parsed[0] != "{":
                text = parsed
                escaped_text = html.escape(text).replace("\n", "<br>")
                inner_parts.append(f"<div style='margin-top:5px;'>{escaped_text}</div>")
            else:
                if type(parsed) == str:
                    parsed = json.loads(parsed)

                if "react" in parsed:
                    react_trace = parsed["react"]
                    for i, item in enumerate(react_trace):
                        
                        if "thought" in item:
                            inner_parts.append(f"💭 <strong>Thought:</strong> {html.escape(item['thought'])}")
                        elif "function" in item:
                            function = html.escape(item.get("function", "unknown"))
                            params = item.get("parameters", {}).get("parameters", {})

                            lines = [f"⚙️ <strong>Action:</strong> {function}"]

                            if params:
                                param_lines = []
                                for key, value in params.items():
                                    label = key.replace("_", " ").capitalize()

                                    if isinstance(value, list):
                                        formatted = ", ".join(
                                            f"<code style='padding:2px 4px; border-radius:4px; background:none; color:inherit'>{html.escape(str(v))}</code>"
                                            for v in value
                                        )
                                        value_str = f"[{formatted}]"
                                    elif isinstance(value, (int, float)):
                                        value_str = str(value)
                                    else:
                                        value_str = html.escape(str(value))

                                    param_lines.append(f"<li><strong>{label}:</strong> {value_str}</li>")

                                param_block = "<ul style='margin-left: 1.5em; margin-top: 0.3em'>" + "".join(param_lines) + "</ul>"
                                lines.append(param_block)

                            inner_parts.append("<br>".join(lines))
                    
                if "response" in parsed:
                    text = parsed["response"].strip()
                    if text:
                        escaped_text = html.escape(text).replace("\n", "<br>")
                        inner_parts.append(f"<div style='margin-top:5px;'><strong>Response:</strong>: {escaped_text}</div>")

                if "tool_response" in parsed:
                    text = parsed["tool_response"].strip()
                    tool_id = str(parsed['tool_call_id']).strip()
                    tool_call_note = f" <span style='color:#666; font-size:0.85em;'>(Tool Call ID: <code>{tool_id}</code>)</span>"

                    if text:
                        if text.strip().startswith("<"):
                            formatted_text = f"<pre style='background:#f8f8f8; padding:8px; border-radius:6px; overflow:auto; font-family:monospace; font-size:0.9em'><code style='background:none; color:inherit'>{html.escape(text)}</code></pre>"
                        else:
                            formatted_text = html.escape(text).replace("\n", "<br>")
                        inner_parts.append(f"<div style='margin-top:5px;'><strong>Response:</strong><br>{formatted_text}</div>")

            content_block = "<br>".join(inner_parts)
            html_parts.append(
                f"<div style='background:{bg_color}; color:{text_color}; border:{border}; border-radius:{border_radius}; padding:{padding}; margin:{margin};'>"
                f"<strong>{role}{tool_call_note}:</strong><br>{content_block}</div>"
            )

        html_parts.append("</div>")
        return HTML("".join(html_parts))
    




    def render_message_content(self, parsed):
        inner_parts = []

        if type(parsed) == str and parsed[0] != "{":
            text = parsed
            escaped_text = html.escape(text).replace("\n", "<br>")
            inner_parts.append(f"<div style='margin-top:5px;'>{escaped_text}</div>")
        else:
            if type(parsed) == str:
                parsed = json.loads(parsed)

            if "react" in parsed:
                react_trace = parsed["react"]
                for i, item in enumerate(react_trace):
                    
                    if "thought" in item:
                        inner_parts.append(f"💭 <strong>Thought:</strong> {html.escape(item['thought'])}")
                    elif "function" in item:
                        function = html.escape(item.get("function", "unknown"))
                        params = item.get("parameters", {}).get("parameters", {})

                        lines = [f"⚙️ <strong>Action:</strong> {function}"]

                        if params:
                            param_lines = []
                            for key, value in params.items():
                                label = key.replace("_", " ").capitalize()

                                if isinstance(value, list):
                                    formatted = ", ".join(
                                        f"<code style='padding:2px 4px; border-radius:4px; background:none; color:inherit'>{html.escape(str(v))}</code>"
                                        for v in value
                                    )
                                    value_str = f"[{formatted}]"
                                elif isinstance(value, (int, float)):
                                    value_str = str(value)
                                else:
                                    value_str = html.escape(str(value))

                                param_lines.append(f"<li><strong>{label}:</strong> {value_str}</li>")

                            param_block = "<ul style='margin-left: 1.5em; margin-top: 0.3em'>" + "".join(param_lines) + "</ul>"
                            lines.append(param_block)

                        inner_parts.append("<br>".join(lines))
                
            if "response" in parsed:
                text = parsed["response"].strip()
                if text:
                    escaped_text = html.escape(text).replace("\n", "<br>")
                    inner_parts.append(f"<div style='margin-top:5px;'><strong>Response:</strong>: {escaped_text}</div>")

            if "tool_response" in parsed:
                text = parsed["tool_response"].strip()
                tool_id = str(parsed['tool_call_id']).strip()
                tool_call_note = f" <span style='color:#666; font-size:0.85em;'>(Tool Call ID: <code>{tool_id}</code>)</span>"

                if text:
                    if text.strip().startswith("<"):
                        formatted_text = f"<pre style='background:#f8f8f8; padding:8px; border-radius:6px; overflow:auto; font-family:monospace; font-size:0.9em'><code style='background:none; color:inherit'>{html.escape(text)}</code></pre>"
                    else:
                        formatted_text = html.escape(text).replace("\n", "<br>")
                    inner_parts.append(f"<div style='margin-top:5px;'><strong>Response:</strong><br>{formatted_text}</div>")

        return "<br>".join(inner_parts)



    def render_message(self, message, st=False):
        try:
            parsed = json.dumps(message)
            parsed = json.loads(parsed)["content"]['text']
        except (KeyError, json.JSONDecodeError):
            return
        
        
        role = message.get("role", "Unknown").capitalize()
        tool_call_note = ""
        '''
        if "tool_call_id" in message:
            tool_id = message["tool_call_id"]
            tool_call_note = f" <span style='color:#666; font-size:0.85em;'>(Tool Call ID: <code>{tool_id}</code>)</span>"
        '''
        bg_color = "#e0f7fa" if role == "Assistant" else "#f8f9fa"
        text_color = "#111"
        border = "1px solid #ccc"
        border_radius = "10px"
        padding = "10px"
        margin = "10px 0"

        html_parts = []

        content_block = self.render_message_content(parsed)

        html_parts.append(
            f"<div style='background:{bg_color}; color:{text_color}; border:{border}; border-radius:{border_radius}; padding:{padding}; margin:{margin};'>"
            f"<strong>{role}{tool_call_note}:</strong><br>{content_block}</div>"
        )
        return html_parts
        

    def render_chat_html_new(self, messages=None):
        from IPython.display import HTML

        if messages is None:
            messages = self.chat.messages

        html_parts = ["<div style='font-family:Arial, sans-serif; line-height:1.6;'>"]

        for message in messages:
            html_parts.append(self.render_message(message))

        html_parts.append("</div>")
        return HTML("".join(html_parts))