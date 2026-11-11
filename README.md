# voice-agent-probe

**Phone your voice agent and find out how fast it really is, as a caller hears it.**

Voice agents get tested in the dashboard, over WebRTC, or with logs from the
server. Callers hear something else: the phone network both ways, the
telephony provider's buffering, and your endpointing settings, all stacked
together. "Sub-second" on a laptop can turn into three or four seconds on a
real phone line.

This tool places a real phone call to your agent, has a short scripted
conversation with it, and times everything from the audio as it arrives:

| Measure | What it tells you |
|---|---|
| First words after answering | Does the caller sit in silence after pickup? |
| Reply to a simple question | The gap a caller actually waits through, network included |
| Reply to a longer question | Whether harder turns are slower |
| When talked over | Does it stop when interrupted, and how fast? |
| Started talking while we were mid-sentence | Is endpointing cutting callers off? |
| Longest time talking without a break | Monologues that callers can't get a word into |

Each call also saves a **stereo recording** (agent on the left, caller on the
right, lined up in time), so you can hear every gap the numbers describe.

## A sample report

From a test call over EnableX, with a person on a phone playing the agent to
check the numbers against the recording:

```
| Answered the call                          | yes                   |
| First words after answering                | 1.28 s                |
| Greeting length                            | 3.19 s                |
| Reply to a simple question after           | 1.88 s                |
| Reply to a longer question after           | 1.69 s                |
| When talked over                           | stopped, after 1.27 s |
| Started talking while we were mid-sentence | 0                     |
| Call length                                | 20.72 s               |
```

## Run it

You need Python 3.10+, an account with a telephony provider that can place
calls (Twilio or EnableX for now), and [ngrok](https://ngrok.com) or any other
public tunnel to your machine.

```bash
git clone https://github.com/Rehansanjay/voice-agent-probe
cd voice-agent-probe
pip install -r requirements.txt
cp .env.example .env        # fill in your provider's credentials
```

In one terminal:

```bash
ngrok http 8765
```

In another:

```bash
python -m probe.call --to +14155550100 --label my-agent
```

The report prints when the call ends, and is saved with the recording under
`results/<when>_<label>.{md,json,wav}`.

Call **your own agent**, or one whose owner has published the number as a
demo line. Each call is short, under a minute.

## Compare two versions of your agent

One call is one sample, so for a decision, make a few calls under one label per
version, then compare:

```bash
python -m probe.call --to +14155550100 --label v1 --calls 3
# deploy the change
python -m probe.call --to +14155550100 --label v2 --calls 3
python -m probe.compare v1 v2
```

Real output from the rehearsal below, where v2 is the fake agent changed to
reply 1 s later (one call each):

```
|                                      | v1      | v2      | Change                  |
| Calls answered                       | 1 of 1  | 1 of 1  |                         |
| First words after answering          | 0.85 s  | 0.85 s  | +0.00 s, about the same |
| Reply to a simple question           | 1.74 s  | 2.59 s  | +0.85 s, slower         |
| Reply to a longer question           | 1.49 s  | 2.50 s  | +1.01 s, slower         |
| Stopped after being talked over      | 0.64 s  | 0.62 s  | -0.02 s, about the same |
| Longest time talking without a break | 3.16 s  | 3.33 s  | +0.17 s, about the same |
| Stopped when talked over             | 1 of 1  | 1 of 1  |                         |
| Talked over the caller               | 0 times | 0 times |                         |
```

Times are medians, with the range beside them. A change under 0.25 s is
reported as "about the same", since calls vary that much on their own.
`python -m probe.compare v1` on its own summarises one label's calls.

### Providers

| `--provider` | Notes |
|---|---|
| `twilio` | Needs an upgraded account: trial calls play a notice and wait for a key press, which an agent never gives. |
| `enablex` | Trial calls are cut at 45 s; the script stops at 40 s to fit. |

Both stream the call as 8 kHz μ-law over a websocket. Adding a provider
(Telnyx, Plivo, Exotel and so on) means one small class in
[`probe/providers.py`](probe/providers.py): how to dial, how to hang up, and
the shape of its media messages.

The live calls so far went through EnableX. The Twilio path follows Twilio's
Media Streams format and passes the rehearsal below; if you run it against a
real Twilio call, I'd like to hear how it went.

## How it measures

- **Speech detection.** Silero VAD scores every 32 ms of incoming audio. A
  window counts as speech only if it is also loud enough, because phone-line
  hum can fool Silero on its own. Three speech windows in a row start a turn,
  and 0.8 s of silence ends it.
- **Times are arrival times on your machine.** A reply time is measured from
  the moment our question finished playing to the moment the agent's voice
  arrived back. That is the gap a caller sits through, including the network
  both ways.
- **It never talks over the agent on purpose,** except in the barge-in test.
  It waits for the agent to finish before asking the next question, so slow
  answers aren't blamed on rude timing.
- **The script.** It waits for a greeting, asks "Hi, what can you help me
  with?", then asks for a step-by-step explanation. 1.5 s into that answer, it
  says "Sorry, can I stop you there for a second?" and waits up to 4 s for the
  agent to stop. The lines are recordings in [`probe/lines/`](probe/lines/);
  swap in your own (8 kHz 16-bit mono wav), or edit the text and run
  `python -m probe.make_lines`.
- **Transcript.** If `DEEPGRAM_API_KEY` is set, the report also says what the
  agent said.

## Checked against a fake agent with known timing

```bash
python -m tests.rehearse
```

This plays a stand-in agent over each provider's wire format. The agent
greets, answers and stops when interrupted on a fixed schedule. No call is
placed and no account is needed. Measured times land within 0.15 s of the
built-in ones. It then runs a second version of the fake that replies 1 s
later, and checks that `probe.compare` reports exactly that: both replies
slower, and the greeting unchanged. Add `--calls 2` to check back-to-back
calls too.

The barge-in stop reads about 0.25 s early. That is because it marks the last
moment of speech, and the fake's final syllables before it stops are partly
quiet.

## Limits

- One call is one sample. Network and model latency vary from call to call,
  so run a few before drawing conclusions.
- The script is in English and fixed. An agent that leads the caller through
  a flow of its own may answer something other than the question asked; the
  timings still hold.
- It measures behaviour, not correctness. It does not judge whether the
  answers were right.

## Need more than a few calls?

I'm Rehan; I build and debug voice agents on Pipecat and LiveKit. I also run
paid tests: 20 calls across different times of day, a recording of every
call, and a written list of what to change in your endpointing, interruption
and telephony settings to close the gaps. Open an issue or email
rehansanjay28@gmail.com.

MIT licensed.
