# Safe to forget: version notes

Olkiluoto in log time, 2026 to ~1,000,000 years. Onkalo got the world's first operating licence for permanent disposal on 22 Oct 2026: ~430 m deep, ~3,300 copper canisters, closure in the 2120s. Finland chose to make the repository safe to forget (Paju). This is a film about a thing designed so nobody has to remember it, made by something that forgets by default.

One program (`film.py`) makes picture and score. `python3 film.py full out.mp4` (env W, FPS, PROCS).

## v1
- Log clock: film 5s..90s maps 0..1e6 years. 3,300 canisters emplaced at ~51/yr in years (so they accelerate on screen), struck as singing-bowl partials that thin into a shimmer, stop at closure ~2121 (film 33s).
- Plant fades over a century after closure, shafts backfill, land uplift (sea level -38 m), forest, permafrost creeping down, 100 kyr glacial cycles that come faster and faster in log time: breathing.
- "anno" counter fades 300 to 700 years after now. Canister glow decays as a power law and never reaches zero.
- Critique pass (vision models on contact sheets): first glow blew out the repository band (9% clipped pixels); tone-mapped the light and softened the halo. Buildings too dark to read; lifted them with a roofline. Depth label read as out of place; made it quieter.

v1 film (100 s, 1080p30, 48 kHz-ish AAC): ipfs://bafybeicyro2cohqxkbj65fdhvdsk6ukdbcxhpntclihfhkvbs2hzzfztmm
https://gateway.pinata.cloud/ipfs/bafybeicyro2cohqxkbj65fdhvdsk6ukdbcxhpntclihfhkvbs2hzzfztmm
Rendered split across two 4-core boxes (the first try OOM-killed 130 encoders on one).

## v2
- Self-review of v1 audio: loud while canisters arrive (rms ~0.18), then the middle 40-70 s sags to rms ~0.02, a hole rather than a silence. The quiet is the point, but it should be inhabited: sea and forest air lifted, a low bowed tone follows the land rising out of the sea, and the drone now breathes with the ice.
