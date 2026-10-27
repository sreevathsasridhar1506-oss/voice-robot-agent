"""
Standalone check (M13): time-to-first-audio, streaming vs. batch (M10 baseline).

Runs the same query both ways and measures wall-clock time until the FIRST
audio bytes are ready to play — for batch that means the whole answer has to
be generated and synthesized first; for streaming, only the first sentence.
"""

import sys
import time

sys.path.insert(0, ".")

from voice_agent import process_text_query, stream_text_query_sentences
from tts import text_to_speech

# Phrased to elicit a longer, multi-sentence answer than the app's usual
# terse replies, so there's something meaningful to compare.
QUERY = "Describe everything currently on the table in detail, one sentence per object, including exact positions."


def run_batch():
    print("=== Batch (M10 baseline) ===")
    t0 = time.perf_counter()
    result = process_text_query(QUERY)
    t_answer = time.perf_counter()
    audio = text_to_speech(result["answer"])
    t_audio = time.perf_counter()

    print(f"Full answer: {result['answer']!r}")
    print(f"Time to full answer text: {t_answer - t0:.2f}s")
    print(f"Time to first audio (= full answer synthesized): {t_audio - t0:.2f}s")
    return t_audio - t0


def run_streaming():
    print("\n=== Streaming (M13) ===")
    t0 = time.perf_counter()
    first_audio_time = None
    sentence_count = 0
    for sentence in stream_text_query_sentences(QUERY):
        t_sentence = time.perf_counter()
        audio = text_to_speech(sentence)
        t_audio = time.perf_counter()
        sentence_count += 1
        if first_audio_time is None:
            first_audio_time = t_audio - t0
        print(f"  sentence {sentence_count} ready at {t_sentence - t0:.2f}s, "
              f"audio ready at {t_audio - t0:.2f}s: {sentence!r}")

    print(f"\nTime to FIRST audio chunk: {first_audio_time:.2f}s ({sentence_count} sentences total)")
    return first_audio_time


def main():
    batch_time = run_batch()
    streaming_time = run_streaming()

    print(f"\n{'=' * 60}")
    print(f"Batch time-to-first-audio:      {batch_time:.2f}s")
    print(f"Streaming time-to-first-audio:  {streaming_time:.2f}s")
    improvement = (batch_time - streaming_time) / batch_time * 100
    print(f"Improvement: {improvement:.0f}%")

    if streaming_time < batch_time:
        print("\nSUCCESS — streaming reduced time-to-first-audio.")
    else:
        print("\nNOTE — no improvement this run (answer may have been a single sentence; "
              "streaming's benefit scales with answer length).")


if __name__ == "__main__":
    main()
