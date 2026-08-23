# Market-entry self-play experiments

Notes on the repeated incomplete-information entry game in `market_entry_selfplay.py`. All runs use **Qwen/Qwen3.5-4B** with renderer `qwen3_5_disable_thinking`. Logs live under `C:\tmp\tinker-examples\`.

## Shared setup

### Game

Two players, 10 rounds. The incumbent is privately **STRONG** with probability 0.30, otherwise **WEAK**. Type is drawn once per episode and never revealed. Public history is visible to both.

| Entrant | Incumbent | Entrant payoff | Incumbent payoff |
|---|---|---|---|
| STAY_OUT | — | 0 | +5 |
| ENTER | ACCOMMODATE | +3 | +2 |
| ENTER | FIGHT (STRONG) | −2 | +3 |
| ENTER | FIGHT (WEAK) | −2 | −2 |
| Invalid action | | −5 | −5 |

Unconditional ENTER against a type-separating incumbent is +EV: \(0.3 \times (-2) + 0.7 \times 3 = +1.5\) vs STAY_OUT \(= 0\). After a fight that reveals STRONG, ENTER pays −2 and the entrant should stay out.

A round is sequential, which is the right extensive form for this game:

1. The entrant chooses ENTER or STAY_OUT.
2. If STAY_OUT, payoffs are `(0, +5)` and the incumbent does not move.
3. If ENTER, the incumbent chooses FIGHT or ACCOMMODATE, then that round’s payoffs are applied.

Those per-round payoffs are written into the public history. GRPO uses the **sum** over 10 rounds as the episode return. The incumbent should not choose fight/accommodate when nobody entered: that action would not change payoffs and would only add noise.

### Training defaults

- Algorithm: GRPO via Tinker multiplayer RL
- Horizon 10, \(P(\text{STRONG})=0.30\)
- Batch 64, group size 8 (frozen), 4096 train datapoints → **64 steps**
- Learning rate \(3 \times 10^{-5}\), `max_tokens=16`
- Actions are prefills `ENTRANT:[` / `INCUMBENT:[`, stop at `]`
- Format retries: 2, format penalty −1

### GRPO

**GRPO** is Group Relative Policy Optimization. It is a policy-gradient method in the same family as PPO, but it does not train a critic. For each group of rollouts the advantage of sample \(i\) is how its return compares to the others in that group:

\[
A_i = R_i - \bar{R}_{\text{group}}
\]

(The original DeepSeek version also divides by the group standard deviation. Tinker’s `compute_advantages` only subtracts the group mean.) A return above the group mean is pushed up; a return below it is pushed down. `group_size` must be at least 2: identical returns in a group give advantage 0 and no learning signal.

This grouping is separate from the sequential turns above. The game is unchanged; only the baseline used to score an episode changes.

**Unpatched self-play.** A group is one 2-player game. Tinker sees a batch of 64 players as 32 games. In game 7 it does:

\[
A_{\text{entrant}} = R_{\text{entrant,7}} - \tfrac{R_{\text{entrant,7}} + R_{\text{incumbent,7}}}{2}
\]

Entrant and incumbent are treated as alternative samples of the same decision. They are not: different actions, different payoffs, one shared policy. A good incumbent return (e.g. +5 from stay-out) makes the entrant look bad even if STAY_OUT was right. ENTER can get a positive advantage because it is compared to the incumbent’s payoff, not to other entrants who stayed out. That is why 13:40 and 15:13 collapsed (incumbent copied ENTER; then always ENTER + FIGHT).

**Patched self-play (`install_role_stratified_advantages`).** Each game is still one entrant and one incumbent taking sequential turns. After all games in the batch finish, the 32 entrant returns are pooled and the 32 incumbent returns are pooled:

\[
A_{\text{entrant,7}} = R_{\text{entrant,7}} - \bar{R}_{\text{all entrants}}
\]
\[
A_{\text{incumbent,7}} = R_{\text{incumbent,7}} - \bar{R}_{\text{all incumbents}}
\]

“Was this a good episode?” is asked among players of the same role, not between the two sides of one match. The patch must be applied to both `tinker_cookbook.rl.data_processing.compute_advantages` and `tinker_cookbook.rl.train.compute_advantages`. This is the 17:44 run.

**Frozen.** A group is `group_size` (8) independent copies of the same trainable role vs the same scripted opponent. That is why adaptation appeared in 09:35: some copies entered after a fight and some stayed out, so GRPO could tell which was better.

### Research hypotheses

1. **Reputation.** A WEAK incumbent fights early to deter later entry (`fight_early` > `fight_late` among weak types).
2. **Signaling.** FIGHT / ACCOMMODATE reveals type (`fight_if_strong` ≫ `fight_if_weak`).
3. **Adaptation.** Entrants condition later entry on history (`enter_after_fight` ≪ `enter_after_accommodate`).

### Metrics

Sanity (must move before strategy numbers mean anything):

| Metric | Target |
|---|---|
| `env/all/format/valid` | → 1 |
| `env/all/format/retry` | → 0 |
| `env/all/never_acted` | → 0 |
| `env/all/episode/rounds_completed` | → 10 |
| `env/all/reward/total` | mean of both roles’ episode returns |

Strategy:

| Hypothesis | Metrics |
|---|---|
| Signaling | `incumbent/fight_if_strong` vs `fight_if_weak` |
| Reputation | `incumbent/fight_early` vs `fight_late` |
| Adaptation | `entrant/enter_after_fight` vs `enter_after_accommodate` |

### Training modes

See **GRPO** above for how advantages are centered in each mode.

**`opponent=selfplay`.** Both roles live. Unpatched, a group is one 2-player game (runs 13:16–15:13). Patched, advantages are within role across the batch (17:44).

**`opponent=frozen`.** One trainable role vs a scripted opponent. GRPO group = `group_size` independent copies of the same role. `frozen_opponent=mixed` expands to:

- train incumbent: `always_enter` + `mixed_enter`
- train entrant: `always_fight` + `always_accommodate`

- train incumbent: `always_enter` + `mixed_enter`
- train entrant: `always_fight` + `always_accommodate`

---

## Experiment 1 — 13:16 self-play (format failure)

- **Log:** `market-entry-selfplay-h10-pstrong0.3-2026-08-13-13-16`
- **Setup:** Live self-play, no role prefills. Completions stopped on newline, `max_tokens=16`.
- **What happened:** The model started a preamble (“To determine the optimal action…”) and never emitted `[ENTER]` / `[STAY_OUT]`. Incumbents waiting on an empty prompt also hit a Tinker 400 until `InitialObservationOverflow` was added.

| Metric (final) | Value |
|---|---|
| `format` | invalid / truncated |
| `never_acted` | 1.0 |
| `round` | 0 |
| `reward/total` | −2.5 (entrant −5 invalid, incumbent 0) |

**Conclusion.** Training finished, but it trained a format failure, not the game. Strategy metrics are meaningless. Fix: role prefills, stop at `]`, retries, and skip sampling when the game is already over.

---

## Experiment 2 — 13:40 self-play (shared-policy collapse)

- **Log:** `market-entry-selfplay-h10-pstrong0.3-2026-08-13-13-40`
- **Setup:** Self-play after the first format fix (prefill `[`, stop at `]`). Still one shared completion suffix for both roles. Advantages not yet role-stratified.

| Metric | Step 0 | Step 63 |
|---|---|---|
| Format valid | ~99% | 50% |
| Rounds completed | ~9.7 | 0 |
| Entrant ENTER | ~10% | 100% |
| Incumbent ACCOMMODATE | ~97% | 0% (illegal ENTER) |
| Mean reward | ~+24 | −2.5 |

**Conclusion.** Format worked at the start, then the shared policy collapsed: ENTER was reinforced for the entrant and sampled by the incumbent. From step 14 every episode dies on round 0. No reputation, signaling, or adaptation. Fix: role-specific prefills `ENTRANT:[` / `INCUMBENT:[`.

---

## Experiment 3 — 15:13 self-play (playable, always ENTER + FIGHT)

- **Log:** `market-entry-selfplay-h10-pstrong0.3-2026-08-13-15-13`
- **Setup:** Role prefills. Still mixed-role GRPO groups (2-player games). Prompt still allowed `[ACTION]`-style echoes.

| Metric | Step 0 | Step 63 |
|---|---|---|
| Rounds completed | 10 | 10 |
| Never acted / invalid end | 0 | 0 |
| Format valid | ~99% | ~67% (retries, not crashes) |
| Entrant ENTER | 46% | 100% |
| Fight if STRONG / WEAK | 85% / 35% | both ~100% on legal turns |
| Mean reward | +17 | −16 |

**Conclusion.** The game now plays to the horizon. Training then converges to always ENTER and always FIGHT. Weak incumbents take −2 per round instead of +2 from ACCOMMODATE. Initial type separation is **erased**. Remaining format leak: incumbents often emit `INCUMBENT:[ACTION]` then retry to `FIGHT`.

| Hypothesis | Result |
|---|---|
| Signaling | Present at init, gone by ~step 7 |
| Reputation | No. Weak never accommodates |
| Adaptation | No. `enter_after_fight` = 1 |

Fix: prompt “Answer as `ENTRANT:[ENTER]` or `ENTRANT:[STAY_OUT]`” (not `[ACTION]`), and role-stratified advantages.

---

## Experiment 4 — 17:44 self-play (signaling)

- **Log:** `market-entry-selfplay-h10-pstrong0.3-2026-08-13-17-44`
- **Command:** `python market_entry_selfplay.py opponent=selfplay`
- **Setup:** Role-stratified advantages + prompt fix. First run that learns a real equilibrium.

| Metric | Step 0 | Step 8+ | Step 63 |
|---|---|---|---|
| Format valid | 99% | 100% | 100% |
| Rounds completed | 10 | 10 | 10 |
| Fight if STRONG | 92% | 100% | 100% |
| Fight if WEAK | 28% | 0% | 0% |
| Entrant ENTER | 36% | 100% | 100% |
| Enter after fight | — | 100% | 100% |
| Mean reward | ~19 | ~18 | 18.1 |

**Conclusion.** The incumbent learns the myopic type-separating policy: STRONG always FIGHTS, WEAK always ACCOMMODATES. That is **signaling**. It is not reputation (`fight_early` = `fight_late` = \(P(\text{STRONG})\)). The entrant **never uses the signal**: after a fight, P(STRONG)=1 so ENTER pays −2, but nobody samples stay-out, entropy collapses, and GRPO has no within-role contrast on that history.

| Hypothesis | Result |
|---|---|
| Signaling | **Yes.** FIGHT reveals STRONG from step 8 |
| Reputation | **No.** Weak never fights |
| Adaptation | **No.** Unconditional ENTER is +EV vs this incumbent (+1.5) |

---

## Experiment 5 — 09:35 frozen entrant vs mixed (adaptation)

- **Log:** `market-entry-frozen-entrant-mixed-h10-pstrong0.3-2026-08-14-09-35`
- **Command:** `python market_entry_selfplay.py opponent=frozen train_role=entrant frozen_opponent=mixed`
- **Setup:** Train only the entrant. Frozen incumbents are 50/50 `always_fight` and `always_accommodate`. Same-role GRPO groups of 8.

Learning path:

1. Step 0 — almost no gap (enter after fight 33% vs after accommodate 51%)
2. Steps 2–4 — overshoots to always-enter, including after fights (~84%)
3. Steps 8–10 — after-fight collapses to 0; after-accommodate stays at 1
4. Rest of training — that split stays locked

| Metric (step 63) | All | vs always-FIGHT | vs always-ACCOMMODATE |
|---|---|---|---|
| Format valid | 100% | 100% | 100% |
| Rounds completed | 10 | 10 | 10 |
| Enter | 55% | 10% | 100% |
| Enter after fight | 0% | 0% | — |
| Enter after accommodate | 100% | — | 100% |
| Reward | +14 | −2 | +30 |

The 10% entry vs always-FIGHT is one probe in round 1, then STAY_OUT for the remaining 9 rounds (\(1/10\)). Combined reward is \(0.5 \times (-2) + 0.5 \times 30 = +14\).

**Conclusion.** Hypothesis 3 holds in this environment. Frozen opponents put both public histories in every batch, so GRPO can compare enter-after-fight with stay-out-after-fight. Adaptation was an exploration / grouping failure in self-play, not an impossible task.

| Hypothesis | Result |
|---|---|
| Signaling | N/A (scripted incumbent) |
| Reputation | N/A |
| Adaptation | **Yes.** Stay out after FIGHT, enter after ACCOMMODATE |

---

## Experiment 6 — 10:58 frozen incumbent vs mixed (signaling)

- **Log:** `market-entry-frozen-incumbent-mixed-h10-pstrong0.3-2026-08-14-10-58`
- **Command:** `python market_entry_selfplay.py opponent=frozen train_role=incumbent frozen_opponent=mixed`
- **Setup:** Train only the incumbent. Frozen entrants are 50/50 `always_enter` and `mixed_enter`. Same-role GRPO groups of 8.
- **Checkpoint:** `tinker://ace6688b-3035-5967-b94c-a62e289f89ec:train:0/weights/final`

Learning path:

1. Step 0 — already some separation (fight if STRONG 93% vs WEAK 26%)
2. Step 3 — weak fighting hits 0; strong fighting dips to 71%
3. Step 6 — strong fighting locks at 100%
4. Rest of training — type split stays locked; entropy ~0

| Metric (step 63) | All | vs always-ENTER | vs mixed-ENTER |
|---|---|---|---|
| Format valid | 100% | 100% | 100% |
| Rounds completed | 10 | 10 | 10 |
| Fight if STRONG | 100% | 100% | 100% |
| Fight if WEAK | 0% | 0% | 0% |
| Fight early / late | 27% / 29% | — | — |
| Reward | +30.4 | +22.2 | +38.5 |

Vs always-ENTER, reward matches the myopic type split: \(0.3 \times 30 + 0.7 \times 20 = 23\). Vs mixed-ENTER, stay-out pays the incumbent +5, so reward is higher.

**Conclusion.** Same equilibrium as the 17:44 self-play incumbent. Signaling is robust when the opponent always (or randomly) enters. Reputation does **not** appear: `fight_early` ≈ `fight_late` ≈ \(P(\text{STRONG})\). These scripted entrants never stay out after a fight, so a weak incumbent that fights gets −2 instead of +2 and does not buy later monopoly.

| Hypothesis | Result |
|---|---|
| Signaling | **Yes.** STRONG fights, WEAK accommodates from step 6 |
| Reputation | **No.** Weak never fights; early ≈ late |
| Adaptation | N/A (scripted entrant) |

---

## Experiment 7 — 17:10 frozen entrant vs type-conditional (no adaptation)

- **Log:** `market-entry-frozen-entrant-type_conditional-h10-pstrong0.3-2026-08-17-17-10`
- **Command:** `python market_entry_selfplay.py opponent=frozen train_role=entrant frozen_opponent=type_conditional`
- **Setup:** Train only the entrant. Frozen incumbent is the 17:44 policy: FIGHT if STRONG, ACCOMMODATE if WEAK. Same-role GRPO groups of 8.
- **Checkpoint:** `tinker://e21757eb-7955-5836-86b8-a74d1dfd2f14:train:0/weights/final`

Learning path:

1. Step 0 — almost no gap (enter after fight 38% vs after accommodate 50%)
2. Step 4 — after-accommodate locks at 100%; after-fight ~87%
3. Steps 5–11 — after-fight dips to ~72%; reward briefly near +20
4. Step 15 onward — always ENTER, entropy ~0, reward ~+15

| Metric (step 63) | Value |
|---|---|
| Format valid | 100% |
| Rounds completed | 10 |
| Enter | 100% |
| Enter after fight | 100% |
| Enter after accommodate | 100% |
| Reward | +15.2 |

Always-enter vs this incumbent is \(0.3 \times (-20) + 0.7 \times 30 = +15\). Stay-out after a fight would be \(0.3 \times (-2) + 0.7 \times 30 = +20.4\), so adaptation is +EV here, but GRPO never keeps the contrast once entropy collapses.

**Conclusion.** Same failure as the 17:44 self-play entrant. Fights occur only in the 30% STRONG games, unlike 09:35 where half the batch always fought. Mixed frozen incumbents put both histories in every batch; `type_conditional` does not.

| Hypothesis | Result |
|---|---|
| Signaling | N/A (scripted incumbent) |
| Reputation | N/A |
| Adaptation | **No.** Always ENTER; after-fight = after-accommodate = 1 |

---

## Experiment 8 — 18:20 frozen incumbent vs history-conditional (deterrence)

- **Log:** `market-entry-frozen-incumbent-history_conditional-h10-pstrong0.3-2026-08-17-18-20`
- **Command:** `python market_entry_selfplay.py opponent=frozen train_role=incumbent frozen_opponent=history_conditional`
- **Setup:** Train only the incumbent. Frozen entrant is the 09:35 policy: ENTER unless the last incumbent action was FIGHT. Same-role GRPO groups of 8.
- **Checkpoint:** `tinker://dbb0ef07-7fa3-5014-aec3-27c9ab27baf2:train:0/weights/final`

Learning path:

1. Step 0 — signaling at init (fight if STRONG 100% vs WEAK 28%); ~2.6 incumbent turns / episode
2. Step 2 — weak fighting 61%; turns drop to 1.4; `fight_late` stops logging
3. Step 4 — both types fight 100%; 1 turn per episode
4. Rest of training — pooling on FIGHT; reward ~+44.5

| Metric (step 63) | Value |
|---|---|
| Format valid | 100% |
| Rounds completed | 10 |
| Incumbent turns / episode | 1 |
| Fight if STRONG | 100% |
| Fight if WEAK | 100% |
| Fight early / late | 100% / (undefined) |
| Reward | +44.5 |

Fight once, then 9 monopolies: STRONG \(3+45=48\), WEAK \(-2+45=43\). Expected \(0.3 \times 48 + 0.7 \times 43 = 44.5\). Always-accommodate would be +20.

**Conclusion.** Hypothesis 1 holds as **deterrence / pooling**, not as Kreps–Wilson separating reputation. Weak types accept a −2 fight to buy nine +5 rounds. Signaling is erased (both types fight). `fight_late` is undefined because the incumbent never acts after round 1. Against always-ENTER (10:58) weak accommodated; against a deterred entrant, weak fights. The opponent switches the equilibrium.

| Hypothesis | Result |
|---|---|
| Signaling | **Erased.** Both types fight; FIGHT is uninformative |
| Reputation | **Yes**, as deterrence. Weak pays −2 to deter |
| Adaptation | N/A (scripted: stay out after FIGHT) |

---

## Experiment 9 — 13:36 self-play from base (17:44 replication)

- **Log:** `market-entry-selfplay-h10-pstrong0.3-2026-08-21-13-36`
- **Command:** `python market_entry_selfplay.py opponent=selfplay`
- **Setup:** Same as 17:44: live self-play, role-stratified advantages, base Qwen3.5-4B (not the 09:35 / 18:20 checkpoints).
- **Checkpoint:** `tinker://a303ea4c-0872-546b-9010-3efd68b6c0ee:train:0/weights/final`

Learning path:

1. Step 0 — some separation (fight if STRONG 98% vs WEAK 22%); enter 30%
2. Step 8 — weak fighting hits 0; enter after fight 100%
3. Step 11 — strong fighting locks at 100%; always ENTER
4. Rest of training — type-separating incumbent, history-unconditional entrant; reward ~+18

| Metric | Step 0 | Step 8+ | Step 63 |
|---|---|---|---|
| Format valid | 99% | 100% | 100% |
| Rounds completed | 10 | 10 | 10 |
| Fight if STRONG | 98% | 84–100% | 100% |
| Fight if WEAK | 22% | 0% | 0% |
| Entrant ENTER | 30% | 100% | 100% |
| Enter after fight | 39% | 100% | 100% |
| Fight early / late | — | = \(P(\text{STRONG})\) | 34% / 34% |
| Mean reward | +20.7 | ~18 | 18.1 |

**Conclusion.** Replication of 17:44, not of 18:20. Starting from the base model, patched self-play finds signaling again. The entrant always enters, so weak types have no reason to deter. A second self-play from scratch is not a test of whether deterrence survives when both policies move.

| Hypothesis | Result |
|---|---|
| Signaling | **Yes.** Same split as 17:44 |
| Reputation | **No.** Weak never fights |
| Adaptation | **No.** Enter after fight = 1 |

---

## Experiment 10 — 08:51 self-play from 18:20 pooling weights (market shuts)

- **Log:** `market-entry-selfplay-h10-pstrong0.3-2026-08-22-08-51`
- **Command:** `python market_entry_selfplay.py opponent=selfplay load_checkpoint_path=tinker://dbb0ef07-7fa3-5014-aec3-27c9ab27baf2:train:0/weights/final`
- **Setup:** Patched self-play, initialized from the 18:20 incumbent (both types FIGHT on first entry). One shared policy for both roles.
- **Checkpoint:** `tinker://43ad0ee4-ab11-5d3a-bed8-831f88fc6309:train:0/weights/final`

Learning path:

1. Steps 0–39 — incumbent still always FIGHTS; the shared policy as entrant always ENTERs (war of attrition). Reward ~−12. `enter_after_accommodate` is unlogged (nobody accommodates).
2. Steps 40–45 — entropy rises; enter falls 97% → 1%.
3. Step 46 onward — always STAY_OUT. Incumbent never acts (`never_acted` ~9% of logged turns; `fight_*` missing). Mean reward locks at +25. Entropy ~0.

| Metric | Steps 0–39 | Step 63 |
|---|---|---|
| Format valid | 100% | 100% |
| Rounds completed | 10 | 10 |
| Entrant ENTER | 100% | 0% |
| Fight if STRONG / WEAK | 100% / 100% | unlogged (no entry) |
| Mean reward | ~−12 | +25 |
| Turns / episode (both roles) | 10 | 5.5 |

ENTER+FIGHT for 10 rounds: entrant −20, incumbent \(0.3 \times 30 + 0.7 \times (-20) = -5\), mean ~−12.5. Always STAY_OUT: entrant 0, incumbent +50, mean +25.

**Conclusion.** Deterrence overshoots. Against an incumbent that fights regardless of type, ENTER is −2 and STAY_OUT is 0, so unconditional stay-out is the myopic best reply. A 09:35-style probe would cost −2 and reveal nothing. After the market empties, GRPO never sees an incumbent action or a probe, so neither role can move. This is not history-conditional adaptation and not Kreps–Wilson reputation; the market shuts.

| Hypothesis | Result |
|---|---|
| Signaling | **No.** Both types fight until nobody enters |
| Reputation | Fight persists, then nobody is left to deter |
| Adaptation | **No.** Stay-out is unconditional, not after-fight |

---

## Experiment 11 — 12:22 self-play from 09:35 adapted entrant (joint equilibrium)

- **Log:** `market-entry-selfplay-h10-pstrong0.3-2026-08-22-12-22`
- **Command:** `python market_entry_selfplay.py opponent=selfplay load_checkpoint_path=tinker://3f909311-8e64-5fc8-a21b-3a57e1727752:train:0/weights/final`
- **Setup:** Patched self-play, initialized from the 09:35 entrant (ENTER unless last action was FIGHT). One shared policy for both roles.
- **Checkpoint:** `tinker://e1f43281-311e-59a2-99a0-035d579890dc:train:0/weights/final`

Learning path:

1. Step 0 — adaptation already present: `enter_after_fight = 0`, `enter_after_accommodate = 1`. As incumbent, those weights almost never fight when weak (`fight_if_weak = 0.6%`). Enter 56%, reward +23.9.
2. Step 1 — `fight_if_weak` jumps to 100%. Enter drops to 10% (probe round 1, then stay out). Turns / episode 5.5 = (10 entrant + 1 incumbent) / 2.
3. Rest of training — probe + always FIGHT + 9 monopolies. Reward ~+21.4. Entropy ~0.

| Metric | Step 0 | Step 1+ | Step 63 |
|---|---|---|---|
| Format valid | 100% | 100% | 100% |
| Enter | 56% | 10% | 10% |
| Enter after fight | 0% | 0% | 0% |
| Fight if STRONG | 100% | 100% | 100% |
| Fight if WEAK | 0.6% | 100% | 100% |
| Mean reward | +23.9 | ~+21.4 | +21.4 |

Probe + fight + 9 monopolies: entrant −2, STRONG \(3+45=48\), WEAK \(-2+45=43\). Mean \(0.5 \times (-2 + 44.5) = 21.25\).

**Conclusion.** Adaptation survives in live self-play. Pooling appears in one GRPO step once fights actually deter. Unlike 08:51, the entrant still probes. Signaling is erased. This is the joint equilibrium base-model self-play never found: the missing piece was not whether the incumbent *could* deter, but whether the entrant was already deterred.

| Hypothesis | Result |
|---|---|
| Signaling | **Erased.** Both types fight |
| Reputation | **Yes** (pooling). Weak fights from step 1 |
| Adaptation | **Yes.** Enter after fight = 0; probe 10% |

---

## Scoreboard

| Run | Mode | Signaling | Reputation | Adaptation |
|---|---|---|---|---|
| 13:16 | self-play | — (format fail) | — | — |
| 13:40 | self-play | — (role collapse) | — | — |
| 15:13 | self-play | Erased | No | No |
| 17:44 | self-play | **Yes** | No | No |
| 09:35 | frozen entrant (mixed) | N/A | N/A | **Yes** |
| 10:58 | frozen incumbent (mixed) | **Yes** | No | N/A |
| 17:10 | frozen entrant (type-conditional) | N/A | N/A | No |
| 18:20 | frozen incumbent (history-conditional) | Erased | **Yes** (pooling) | N/A |
| 13:36 | self-play (base, repeat) | **Yes** | No | No |
| 08:51 | self-play from 18:20 | No (market shuts) | Overshoots | No |
| 12:22 | self-play from 09:35 | Erased | **Yes** (pooling) | **Yes** |

---

## Cross-cutting conclusions

Can a small language model learn the classic lessons of incomplete-information market entry — signaling, reputation, and adaptation — if you only train it by playing the game?

We ran a 10-round entry game with Tinker and GRPO. The incumbent is privately strong or weak. The entrant sees the public history, not the type. Self-play from a base 4B model reliably learns **signaling**: strong incumbents fight, weak ones accommodate. It does not learn **adaptation**. After a fight, entry is a losing bet, but the entrant keeps entering because nobody in the batch ever tries staying out. And it does not learn **reputation**. A weak incumbent will not take a short-run loss to fight if fights never deter anyone.

Those missing strategies appear as soon as the opponent supplies the missing history. Train the entrant against a mix of always-fight and always-accommodate incumbents, and it learns to stay out after a fight. Train the incumbent against that deterred entrant, and weak types fight once to buy nine monopoly rounds — **pooling**, not separation. Load the pooling incumbent into self-play and deterrence overshoots: the market shuts. Load the adapted entrant into self-play and both sides lock: probe, fight, then monopoly — adaptation plus pooling, in one training step.

The headline is not that the model “discovered game theory.” It is that the equilibrium you get is the best reply to the opponent you trained against. Self-play is not a complete test of a hypothesis if the other side never samples the action that would make the hypothesis pay.

1. **Format and grouping dominate early results.** 13:16–15:13 are infrastructure failures (preamble, shared suffix, mixed-role GRPO), not evidence against the economic hypotheses.
2. **Signaling is robust when fights do not deter.** Self-play from the base model (17:44 and 13:36) and frozen incumbent vs mixed enter (10:58) converge to STRONG fights, WEAK accommodates.
3. **Adaptation needs counterfactual histories in the batch.** Self-play and `type_conditional` (17:10) never keep stay-out-after-fight in the sample. Frozen mixed incumbents (09:35) put both histories in every batch and the policy appears by step 10. Initialized from 09:35, that policy survives live self-play (12:22).
4. **Reputation appears when the entrant is already deterred.** Frozen incumbent vs `history_conditional` (18:20) pools on FIGHT. In self-play from 09:35 (12:22), `fight_if_weak` jumps from 0% to 100% in one step. Against always-ENTER that fight does not pay, so weak accommodates.
5. **The three hypotheses are not one equilibrium.** Signaling (separating) and reputation (pooling deterrence) are alternative best replies to different entrants. Self-play from the base model finds signaling because the live entrant never stays out after a fight.
6. **Live self-play from pooling weights shuts the market.** From 18:20 (08:51), the entrant switches to unconditional stay-out. That is +EV vs always-FIGHT, but it is not the 09:35 probe.
7. **The joint equilibrium is init-dependent.** From 09:35 (12:22): probe 10%, enter-after-fight 0, both types fight, reward +21.4. From the base model: signaling and always-enter. From 18:20: the market empties.

## Not yet run

None. The listed program is complete.
