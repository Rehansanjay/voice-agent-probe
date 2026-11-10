# voice-agent-probe

Calls a voice agent's phone number and measures how it behaves, as a caller hears it:

- how long until it speaks after picking up
- how long it takes to answer a question
- whether it stops when talked over, and how fast
- whether it starts talking while the caller is mid-sentence

It places a real call through EnableX, streams the audio, plays a short scripted
conversation, and times the agent's speech with Silero VAD on the audio as it arrives.
Every number includes the phone network both ways, because that is what a caller hears.

```
ngrok http 8765
python -m probe.call --to 91XXXXXXXXXX --label acme-demo
```

Writes `results/<when>_<label>.{md,json,wav}`; the wav is stereo, agent left, caller right.

## Checked against a fake agent with known timing

`python -m tests.rehearse` runs the script against a stand-in agent that greets, answers
and stops when interrupted on a fixed schedule. Three runs in a row land within about
0.1-0.2 s of the built-in timings on every measure.
