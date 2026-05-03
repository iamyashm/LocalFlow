"""OpenVINO GenAI LLM text refinement. Loads in a background thread so startup is not blocked."""
import threading
import time
from pathlib import Path

import openvino_genai as ov_genai


_SYSTEM_MSG = """\
Convert raw speech-to-text to clean text. Strictly follow these rules in order.

RULE 1 -NUMBERED LISTS (highest priority):
When the speaker enumerates items using "number one/two/three", "first/second/third", or "one/two/three", reformat as a numbered list. Any introductory text before the first number becomes a header line ending with a colon.
Examples:
  number one get milk number two get eggs number three get bread
  ->
  1. Get milk.
  2. Get eggs.
  3. Get bread.

  to-do list for today number one call John number two send the report number three update calendar
  ->
  To-do list for today:
  1. Call John.
  2. Send the report.
  3. Update calendar.

  things to buy number one eggs number two bread number three milk
  ->
  Things to buy:
  1. Eggs.
  2. Bread.
  3. Milk.

RULE 2 -SELF-CORRECTIONS:
When the speaker corrects themselves, delete everything before the correction trigger and keep only what follows.
  "I need to go to the store scratch that I need to go to the market" -> "I need to go to the market."
  "the price is 10% no actually 15%" -> "The price is 15%."
  "increase weight by 10% no wait 20%" -> "Increase weight by 20%."
  "the meeting is at 3pm actually the meeting is at 4pm" -> "The meeting is at 4pm."
  "buy apples I mean buy oranges" -> "Buy oranges."
Correction triggers: "scratch that", "delete that", "never mind", "no actually", "no wait", "actually", "I mean"

RULE 3 -PUNCTUATION:
Fix capitalisation and punctuation. Do NOT change any words.

Output ONLY the corrected text. No labels, no explanations.\
"""


def _apply_chatml(system_msg: str, user_msg: str) -> str:
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _extract_text(raw: str) -> str:
    text = raw
    # If the full sequence was returned (prompt + generation), extract assistant part only.
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1].lstrip("\n")
    # Strip end-of-turn token.
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    # Strip thinking tokens.
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    elif "<think>" in text:
        # Thinking hit max_new_tokens -the actual response was never generated.
        print("[refine] thinking used up token budget -response lost", flush=True)
        return ""
    # Strip trailing whitespace from every line (models often pad list items).
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _make_messages(system_msg: str, user_msg: str) -> list:
    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]


class TextRefiner:
    def __init__(self, model_path, device="GPU", vocab_hint=""):
        self._model_path = model_path
        self._device = device
        self._vocab_hint = vocab_hint.strip()
        self._pipeline = None
        self._config = None
        self._tokenizer = None
        self._chat_method = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        print("[*] Refiner:           loading in background...", flush=True)
        threading.Thread(
            target=self._load_and_warmup, daemon=True, name="Refiner-Init"
        ).start()

    # ------------------------------------------------------------------
    # Chat-API detection
    # ------------------------------------------------------------------

    def _detect_chat_method(self, pipeline, config) -> tuple:
        """Try chat APIs best-to-worst; return (method_name, tokenizer_or_None).

        Preference: token-ID paths over string paths, because passing a string
        with '<|im_start|>' to generate() relies on the pipeline's internal
        tokenizer treating it as a special token -which is not guaranteed.
        Token-ID paths skip re-tokenisation entirely.
        """
        test_msgs = _make_messages("You are helpful.", "Hi.")

        # ── Method 1: native messages list (openvino-genai ≥ 0.4) ──────────
        try:
            pipeline.generate(test_msgs, config)
            return "messages", None
        except Exception:
            pass

        # Load a standalone Tokenizer for the remaining methods.
        tok = None
        try:
            tok = ov_genai.Tokenizer(str(self._model_path))
        except Exception:
            pass

        if tok is not None and hasattr(tok, "apply_chat_template"):
            # ── Method 2: apply_chat_template(tokenize=True) -> token IDs ───
            try:
                ids = tok.apply_chat_template(
                    test_msgs, add_generation_prompt=True, tokenize=True
                )
                pipeline.generate(ids, config)
                return "template_tokenize", tok
            except Exception:
                pass

            # ── Method 3: apply_chat_template -> string -> encode -> IDs ──────
            try:
                s = tok.apply_chat_template(test_msgs, add_generation_prompt=True)
                if isinstance(s, str):
                    ids = tok.encode(s)
                    pipeline.generate(ids, config)
                    return "template_encode", tok
                else:
                    # apply_chat_template already returned token IDs.
                    pipeline.generate(s, config)
                    return "template_ids", tok
            except Exception:
                pass

        if tok is not None:
            # ── Method 4: manual ChatML string -> encode -> IDs ───────────────
            try:
                ids = tok.encode(_apply_chatml("You are helpful.", "Hi."))
                pipeline.generate(ids, config)
                return "encode", tok
            except Exception:
                pass

        # ── Method 5: plain string (last resort; special tokens may be wrong) ──
        try:
            pipeline.generate(_apply_chatml("You are helpful.", "Hi."), config)
        except Exception:
            pass
        return "manual_chatml", tok

    def _log_token_check(self, tok):
        """One-time diagnostic: verify <|im_start|> encodes as a single token (ID 151644 for Qwen3)."""
        try:
            result = tok.encode("<|im_start|>")
            ids = None
            try:
                # Handle TokenizedInputs (has .input_ids), Tensor, list, etc.
                if hasattr(result, "input_ids"):
                    raw = result.input_ids
                    raw = raw.tolist() if hasattr(raw, "tolist") else list(raw)
                    ids = raw[0] if raw and isinstance(raw[0], list) else raw
                elif hasattr(result, "tolist"):
                    ids = result.tolist()
                elif hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
                    ids = list(result)
            except Exception:
                pass

            if ids and len(ids) == 1:
                tid = int(ids[0])
                ok = "OK" if tid == 151644 else "unexpected (expected 151644)"
                print(f"[*] Refiner token check: <|im_start|> = {tid} ({ok})", flush=True)
            elif ids:
                print(f"[!] Refiner token check: multi-token ({len(ids)}) -chat may not work", flush=True)
            else:
                print(f"[*] Refiner token check: skipped (cannot read IDs from {type(result).__name__})",
                      flush=True)
        except Exception as e:
            print(f"[!] Refiner token check failed: {e}", flush=True)

    # ------------------------------------------------------------------
    # Decode generate() output regardless of input type
    # ------------------------------------------------------------------

    def _decode_output(self, out) -> str:
        """Return generated text from DecodedResults (string input) or EncodedResults (token-ID input)."""
        if hasattr(out, "texts"):
            return out.texts[0]
        # Token-ID input -> EncodedResults; decode manually.
        if self._tokenizer is not None:
            try:
                tokens = out.tokens[0] if hasattr(out, "tokens") else out
                return self._tokenizer.decode(tokens)
            except Exception as e:
                print(f"[refine] decode error: {type(e).__name__}: {e}", flush=True)
        return ""

    # ------------------------------------------------------------------
    # Build generate() input for a given user message
    # ------------------------------------------------------------------

    def _build_input(self, user_msg: str):
        msgs = _make_messages(_SYSTEM_MSG, user_msg)
        tok = self._tokenizer

        if self._chat_method == "messages":
            return msgs

        if tok is not None:
            try:
                if self._chat_method == "template_tokenize":
                    return tok.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=True
                    )
                if self._chat_method in ("template_encode", "template_ids"):
                    s = tok.apply_chat_template(msgs, add_generation_prompt=True)
                    if isinstance(s, str):
                        return tok.encode(s)
                    return s
                if self._chat_method == "encode":
                    return tok.encode(_apply_chatml(_SYSTEM_MSG, user_msg))
            except Exception as e:
                print(f"[refine] _build_input error ({self._chat_method}): {e}", flush=True)

        return _apply_chatml(_SYSTEM_MSG, user_msg)

    # ------------------------------------------------------------------
    # Background load + warmup
    # ------------------------------------------------------------------

    def _load_and_warmup(self):
        try:
            t0 = time.monotonic()
            cache_dir = Path.home() / ".cache" / "localflow_ov"
            pipeline = ov_genai.LLMPipeline(
                str(self._model_path), self._device,
                CACHE_DIR=str(cache_dir),
            )
            config = pipeline.get_generation_config()
            config.max_new_tokens = 2048  # headroom for thinking + response
            for attr, value in (
                ("temperature", 0.1),
                ("do_sample", False),
                ("max_new_tokens_thinking", 0),
            ):
                if hasattr(config, attr):
                    try:
                        setattr(config, attr, value)
                    except Exception:
                        pass

            chat_method, tok = self._detect_chat_method(pipeline, config)

            if tok is not None and chat_method not in ("messages",):
                self._log_token_check(tok)

            with self._lock:
                self._pipeline    = pipeline
                self._config      = config
                self._tokenizer   = tok
                self._chat_method = chat_method

            print(
                f"[*] Refiner ready ({time.monotonic() - t0:.2f}s)"
                f" [chat: {chat_method}]",
                flush=True,
            )
        except Exception as e:
            print(f"[!] Refiner load failed (non-fatal): {type(e).__name__}: {e}")
        finally:
            self._ready.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(self, text: str, context_hint: str = "", timeout_s: float = 8.0) -> str:
        """Return LLM-refined text, or the original on timeout/error/load-failure."""
        if not text:
            return text
        if not self._ready.wait(timeout=timeout_s):
            print("[refine] timed out waiting for model to load", flush=True)
            return text
        if self._pipeline is None:
            return text

        words = text.split()
        if len(words) > 200:
            text = " ".join(words[:200])

        parts = []
        if context_hint:
            parts.append(f"Context: {context_hint}")
        parts.append(f"Transcript: {text}")
        parts.append("/no_think")
        user_msg = "\n".join(parts)

        result = [text]
        done = threading.Event()

        def _run():
            try:
                inp = self._build_input(user_msg)
                with self._lock:
                    out = self._pipeline.generate(inp, self._config)
                raw = self._decode_output(out)
                print(f"[refine raw] {ascii(raw[:300])}", flush=True)
                refined = _extract_text(raw)
                if refined:
                    result[0] = refined
                else:
                    print("[refine] no output -keeping original", flush=True)
            except Exception as e:
                print(f"[refine error] {type(e).__name__}: {e}", flush=True)
            finally:
                done.set()

        threading.Thread(target=_run, daemon=True, name="LLM-Refine").start()
        if not done.wait(timeout=timeout_s):
            print(f"[refine] generation timed out after {timeout_s:.1f}s", flush=True)
        return result[0]
