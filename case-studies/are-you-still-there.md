# "Sorry, are you still there?" just as the caller starts talking

A case study from my own outbound voice agent, Atlas (Twilio, Deepgram speech-to-text and text-to-speech, an LLM). Two calls, both to my own phone, on 11 November 2026.

## What the caller heard

I picked up a test call and answered the agent's questions normally. Twice, right as I began to reply, the agent said:

> Sorry, are you still there?

The first time, I had just said "Manually." My answer then sat behind the prompt, and the agent's real reply came 5.5 s after I stopped talking. The second time, it cut in as I began the next answer.

To a caller this sounds like the agent wasn't listening, and it lands on the exact moment they decide to engage.

## How I found the cause

The call was recorded with the agent and me on separate channels, so I could line up the speech on each side.

Both prompts came after the agent's two longest replies, which played for about 15 s each. That matched the agent's silence watchdog: after 15 s of nothing from the caller, it asks once whether they're still there, and 15 s after that it ends the call.

The watchdog started counting when a reply finished generating. But the agent sends audio to Twilio faster than real time, and Twilio plays it out at normal speed. So a reply could finish generating while most of it was still waiting to be heard. On a 15 s reply, the 15 s "silence" window had nearly run out by the time the caller heard the last word.

## The fix

- Start the silence countdown when Twilio confirms playback has finished, not when the reply finishes generating. Twilio does this with a `mark` message: you send one after the audio, and Twilio sends it back once everything before it has played.
- While the agent is still talking, keep a backstop of an extra 60 s. If a confirmation is ever lost, the call still can't stay open forever.
- A reply can go out in several pieces, and each piece can send its own mark. Only the last one counts as "finished playing".

About 20 lines in one file.

## The result

I made a second call to the same number and played the caller the same way:

| | Before | After |
|---|---|---|
| Long replies (12-15 s) the caller answered | 2 | 3 |
| "Are you still there?" as the caller replied | 2 | 0 |

**What I haven't tested yet:** the prompt should still play after 15 s of real silence. The code path is the same, but no call has checked it yet.

## If you run a voice agent

Most voice agents send speech audio ahead of real time, because sending it late makes it stutter. If yours does, any timer that starts when a reply is generated is starting too early. That includes "are you still there?" prompts, turn timeouts and call-end timers. Start them from your platform's playback-finished signal instead, such as Twilio's `mark` or Pipecat's `BotStoppedSpeakingFrame`.

You can't see this bug in logs, because every reply looks like it went out fine. It only shows up when you listen to the call from the caller's side.

## What's still slow

The same calls show replies taking 3.0 to 5.5 s after the caller stops talking, with a typical figure around 4 s. That's a separate problem and the next one to fix. I'm not claiming this change made the agent faster.

---

The calls were made and measured with [voice-agent-probe](https://github.com/Rehansanjay/voice-agent-probe). If you'd like the same check on your agent, with recordings and a list of what to change, email rehansanjay28@gmail.com.
