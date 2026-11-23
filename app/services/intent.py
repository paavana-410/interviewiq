"""Conversational Intent Classification for Virtual Placement Interviewer.

Detects meta-conversational utterances such as:
- Asking for a moment / thinking time ("Can I have a moment?", "Give me a second")
- Passing / expressing lack of knowledge ("I don't know", "I shall pass", "Let's skip")
- Asking to repeat / clarify the question ("Could you repeat that?", "What was the question?")
- Microphone / connectivity checks ("Can you hear me?", "Is my mic working?")
- Greetings / pleasantries ("Hello", "Hi there")
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ConversationalIntent(str, Enum):
    ANSWER = "answer"
    NEED_MOMENT = "need_moment"
    PASS = "pass"
    REPEAT = "repeat"
    MIC_CHECK = "mic_check"
    GREETING = "greeting"


@dataclass
class IntentResult:
    intent: ConversationalIntent
    is_conversational: bool
    interviewer_reply: str
    action: str  # "pause_and_wait" | "pass_question" | "repeat_question" | "ready_for_answer" | "proceed_to_evaluation"


_MOMENT_PATTERNS = [
    r"\b(can|could|may)\s+(i|we)\s+(have|get|take)\s+(a\s+)?(moment|minute|second|sec|breath|time|min)\b",
    r"\b(give|give\s+me)\s+(a\s+)?(second|moment|minute|sec|min)\b",
    r"\b(let\s+me|lemme)\s+(think|see|gather|process|ponder|collect)\b",
    r"\b(just\s+a\s+)?(moment|second|minute|sec)\s*(please)?\b",
    r"\b(hold\s+on|wait\s+a\s+(second|moment|minute)|hang\s+on|wait\s+wait)\b",
    r"\bi\s+need\s+(a\s+)?(moment|second|minute|sec|time)\b",
    r"\bneed\s+(a\s+)?(moment|second|minute)\b",
    r"\bthinking\s+time\b",
]

_PASS_PATTERNS = [
    r"\b(i\s+)?(shall|will|can\s+(i|we)|let'?s|please)\s+(pass|skip)\b",
    r"\b(pass|skip)\s*(this|the)?\s*(question)?\b",
    r"\bi\s+don'?t\s+(know|have\s+an\s+answer|recall|remember|have\s+any\s+idea|understand|know\s+what('?s|\s+is)\s+happening|know\s+this\s+is\s+happening)\b",
    r"\b(no\s+idea|not\s+sure|not\s+familiar|haven'?t\s+(heard|worked|studied|learned))\b",
    r"\b(move\s+on|next\s+question)\b",
    r"\b(i\s+would\s+like\s+to\s+pass|let\s+us\s+move\s+to\s+next)\b",
    # Explicit UI-generated pass marker (from the Pass button in interview.html)
    r"candidate\s+passed\s+this\s+question\s+without\s+answering",
]

_REPEAT_PATTERNS = [
    r"\b(repeat|say\s+that\s+again|come\s+again|pardon|what\s+was\s+the\s+question|didn'?t\s+catch|can\s+you\s+repeat)\b",
    r"\b(could\s+you|can\s+you)\s+(say|read|repeat)\s+(it|that|the\s+question)\b",
    r"\bwhat\s+did\s+you\s+say\b",
]

_MIC_PATTERNS = [
    r"\b(can\s+you\s+hear\s+me|am\s+i\s+audible|is\s+my\s+mic\s+working|testing\s+1\s*2|can\s+you\s+hear)\b",
    r"\bhello\s*\?+\s*(can\s+you\s+hear)?\b",
]

_GREETING_PATTERNS = [
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\s*(!|\.)?$",
]


def classify_conversational_intent(transcript: str, question_text: str = "") -> IntentResult:
    """Classify whether the transcript is a meta-conversational remark or an actual answer."""
    clean = transcript.strip().lower()
    clean_no_punct = re.sub(r"[^\w\s]", "", clean)
    words = clean_no_punct.split()
    word_count = len(words)

    # Empty, near-empty, or common silence/noise ASR hallucinations
    _SILENCE_ARTIFACTS = {
        "thank you", "thank you.", "thanks", "thank you for watching", "thank you very much",
        "subtitles by", "watching", "bye", "bye bye", "silence", "music", "applause",
        "you", "so", "um", "uh", "okay", "yeah", "yes", "no"
    }
    if not clean or word_count == 0 or clean_no_punct in _SILENCE_ARTIFACTS or word_count < 3:
        # Check if candidate explicitly passed or asked to repeat even in <= 2 words
        is_pass_short = False
        for p in _PASS_PATTERNS:
            if re.search(p, clean_no_punct):
                is_pass_short = True
                break
        if is_pass_short:
            return IntentResult(
                intent=ConversationalIntent.PASS,
                is_conversational=True,
                interviewer_reply="Understood. Let's move on to the next question.",
                action="pass_question",
            )

        is_repeat_short = False
        for p in _REPEAT_PATTERNS:
            if re.search(p, clean_no_punct):
                is_repeat_short = True
                break
        if is_repeat_short:
            q_mention = f"The question is: {question_text}" if question_text else "Let me repeat the question for you."
            return IntentResult(
                intent=ConversationalIntent.REPEAT,
                is_conversational=True,
                interviewer_reply=f"Certainly! {q_mention}",
                action="repeat_question",
            )

        return IntentResult(
            intent=ConversationalIntent.NEED_MOMENT,
            is_conversational=True,
            interviewer_reply="I didn't hear your response clearly. Please click Start Answer and speak into your microphone.",
            action="pause_and_wait",
        )

    # 1. Check for Repeat request (typically short: <= 15 words)
    if word_count <= 15:
        for p in _REPEAT_PATTERNS:
            if re.search(p, clean_no_punct):
                q_mention = f"The question is: {question_text}" if question_text else "Let me repeat the question for you."
                return IntentResult(
                    intent=ConversationalIntent.REPEAT,
                    is_conversational=True,
                    interviewer_reply=f"Certainly! {q_mention}",
                    action="repeat_question",
                )

    # 2. Check for Pass / Don't know (typically <= 18 words)
    if word_count <= 18:
        for p in _PASS_PATTERNS:
            if re.search(p, clean_no_punct):
                return IntentResult(
                    intent=ConversationalIntent.PASS,
                    is_conversational=True,
                    interviewer_reply="No worries at all! Placement interviews cover broad ground. Let's move ahead to the next question.",
                    action="pass_question",
                )

    # 3. Check for Requesting Time / Moment (typically <= 16 words)
    if word_count <= 16:
        for p in _MOMENT_PATTERNS:
            if re.search(p, clean_no_punct):
                return IntentResult(
                    intent=ConversationalIntent.NEED_MOMENT,
                    is_conversational=True,
                    interviewer_reply="Of course, take all the time you need! Take a breath, and whenever you're ready, click Start Answer.",
                    action="pause_and_wait",
                )

    # 4. Check for Mic check / Audibility
    if word_count <= 12:
        for p in _MIC_PATTERNS:
            if re.search(p, clean_no_punct):
                return IntentResult(
                    intent=ConversationalIntent.MIC_CHECK,
                    is_conversational=True,
                    interviewer_reply="I hear you loud and clear! Whenever you're ready, feel free to start your answer.",
                    action="ready_for_answer",
                )

    # 5. Greeting alone
    if word_count <= 5:
        for p in _GREETING_PATTERNS:
            if re.search(p, clean_no_punct):
                return IntentResult(
                    intent=ConversationalIntent.GREETING,
                    is_conversational=True,
                    interviewer_reply="Hello! Great to have you here. Take a second to review the question on screen, and click Start Answer when you're ready.",
                    action="ready_for_answer",
                )

    # 6. Default: Substantive Answer
    return IntentResult(
        intent=ConversationalIntent.ANSWER,
        is_conversational=False,
        interviewer_reply="",
        action="proceed_to_evaluation",
    )
