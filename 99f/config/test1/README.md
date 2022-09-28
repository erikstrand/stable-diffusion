# Whole Frame img2img2 Tests

## First Batch

### Prompts

The same 4 prompts are used throughout these tests.

0. an underwater paradise, disney film, art by atey ghailan, stephen bliss, makoto shinkai, james gilleard, joe fenton, cinematic, tone mapped, intricate, award winning details, realistic background, detailed line art, trending on artstation
1. an underwater wasteland, floating plastic refuse and garbage, disney film, art by atey ghailan, stephen bliss, makoto shinkai, james gilleard, joe fenton, cinematic, tone mapped, intricate, award winning details, realistic background, detailed line art, dark ominous lighting, trending on artstation
2. a scene from the little mermaid, disney animation
3. a scene from the little mermaid, disney animation,:0.3 garbage floating underwater:0.7

### Clip 00

- prompt: 0
- seed: same for every frame (2639741)
- strength: 0.01
- length: 251 frames

No noticeable modifications at this strength.

### Clip 01

- prompt: 0
- seed: same for every frame (2639741)
- strength: 0.05
- length: 251 frames

Very minor changes, not noticeable as anything but slight corruption.

### Clip 02

- prompt: 0
- seed: same for every frame (2639741)
- strength: 0.1
- length: 251 frames

Some semi-coherent changes start to appear (e.g. Sebastian glitching to being a cyclops for a bit)

### Clip 03

- prompt: 0
- seed: same for every frame (2639741)
- strength: 0.2
- length: 251 frames

Getting pretty trippy. Still a lot of the changes just read as corruption.

### Clip 04

- prompt: 0
- seed: same for every frame (2639741)
- strength: 0.3
- length: 251 frames

Ariel and Sebastian start glitching into completely different forms.

### Clip 05

- prompt: 0
- seed: same for every frame (2639741)
- strength: 0.4
- length: 251 frames

Characters often appear in different forms, e.g. Ariel as abstract art and Sebastian as a red fish.

### Clip 06

- prompt: 0
- seed: same for every frame (2639741)
- strength: 0.5
- length: 251 frames

Characters are themselves a minority of the time. Ariel talking to Sebastian as a fish is awesome.

### Clip 07

- prompt: 1
- seed: same for every frame (2639741)
- strength: 0.2
- length: 251 frames

At this strength, results are basically indistinguishable from the other prompt.

### Clip 08

- prompt: 1
- seed: same for every frame (2639741)
- strength: 0.5
- length: 251 frames

Still extremely similar. We should try vastly different prompts. Ariel talking to Sebastian the fish
might be slightly better here.

### Clip 09

- prompt: 2
- seed: same for every frame (2639741)
- strength: 0.2
- length: 251 frames

Even with a completely different prompt, the distortions feel very similar to the others at this strength.

### Clip 10

- prompt: 3
- seed: same for every frame (2639741)
- strength: 0.2
- length: 251 frames

This feels the same as the last.

### Clip 11

- prompt: interpolates between prompts 0 and 1
- seed: same for every frame (9087143)
- strength: interpolates between 0.0 and 0.5
- length: 751 frames

Modulating the strength is a very fun effect. We'll probably want to fine tune it a good deal, so we
can dial it up aggressively when things are more static, and tone it down when there is a lot of motion.

### Clip 12

- prompt: interpolates between prompts 2 and 3
- seed: same for every frame (9087143)
- strength: interpolates between 0.0 and 0.5
- length: 751 frames

Ariel's shape changing seems somewhat more in style with this prompt. If we dialed up the strength
more, I think she would morph between different disney princesses.

### Clip 13

- prompt: 0
- seed: different every frame
- strength: 0.1
- length: 251 frames

We get the Johnny Cash effect, but it's way too fast. I think it would work better to process every
third frame or so, and interpolate between those.

### Clip 14

- prompt: 0
- seed: different every frame
- strength: 0.2
- length: 251 frames

Stronger effect, but somehow feels more coherent. Could be fine for an intentionally wild section.

### Clip 15

- prompt: 0
- seed: different every frame
- strength: 0.3
- length: 251 frames

Getting even crazier, but still easy to follow what's going on.

### Clip 16

- prompt: 0
- seed: different every frame
- strength: 0.4
- length: 251 frames

Too much, too fast...

### Clip 17

- prompt: 0
- seed: different every frame
- strength: 0.5
- length: 251 frames

Do not show this to people with epilepsy.
