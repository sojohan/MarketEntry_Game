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

Unconditional ENTER against a type-separating incumbent is $$+EV: \(0.3 \times (-2) + 0.7 \times 3 = +1.5\)$$ vs STAY_OUT \(= 0\). After a fight that reveals STRONG, ENTER pays −2 and the entrant should stay out.

A round is sequential, which is the right extensive form for this game:

1. The entrant chooses ENTER or STAY_OUT.
2. If STAY_OUT, payoffs are `(0, +5)` and the incumbent does not move.
3. If ENTER, the incumbent chooses FIGHT or ACCOMMODATE, then that round’s payoffs are applied.

Those per-round payoffs are written into the public history. GRPO uses the **sum** over 10 rounds as the episode return. The incumbent should not choose fight/accommodate when nobody entered: that action would not change payoffs and would only add noise.

### Training defaults

- Algorithm: GRPO via Tinker multiplayer RL
- Horizon 10, $$\(P(\text{STRONG})=0.30\)$$
- Batch 64, group size 8 (frozen), 4096 train datapoints → **64 steps**
- Learning rate $ \(3 \times 10^{-5}\)$, `max_tokens=16`
- Actions are prefills `ENTRANT:[` / `INCUMBENT:[`, stop at `]`
- Format retries: 2, format penalty −1

### GRPO

**GRPO** is Group Relative Policy Optimization. It is a policy-gradient method in the same family as PPO, but it does not train a critic. For each group of rollouts the advantage of sample \(i\) is how its return compares to the others in that group:
$$
\[
A_i = R_i - \bar{R}_{\text{group}}
\]
$$
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

## Scoreboard

| Run | Mode | Signaling | Reputation | Adaptation |
|---|---|---|---|---|
| 13:16 | self-play | — (format fail) | — | — |
| 13:40 | self-play | — (role collapse) | — | — |
| 15:13 | self-play | Erased | No | No |
| 17:44 | self-play | **Yes** | No | No |
| 09:35 | frozen entrant | N/A | N/A | **Yes** |
| 10:58 | frozen incumbent | **Yes** | No | N/A |

---

## Cross-cutting conclusions

1. **Format and grouping dominate early results.** 13:16–15:13 are infrastructure failures (preamble, shared suffix, mixed-role GRPO), not evidence against the economic hypotheses.
2. **Signaling is robust.** Both self-play (17:44) and frozen incumbent vs mixed enter (10:58) converge to STRONG fights, WEAK accommodates.
3. **Adaptation needs counterfactual histories in the batch.** Self-play never samples stay-out-after-fight against a separating incumbent, so the contrast never appears. Frozen mixed incumbents supply that contrast and the policy appears by step 10 (09:35).
4. **Reputation is still untested.** Frozen incumbent vs `always_enter` / `mixed_enter` cannot produce it, because those opponents are not deterred. It needs a weak incumbent that fights early *and* an entrant that stays out after a fight.

## Not yet run

A history-conditional frozen entrant (enter unless the last response was FIGHT) does not exist yet. Accepted `frozen_opponent` values are only `mixed`, `always_enter`, `always_stay_out`, `mixed_enter`, `always_fight`, `always_accommodate`, `mixed_respond`, `type_conditional`.

Once that kind exists:

```text
python market_entry_selfplay.py opponent=frozen train_role=incumbent frozen_opponent=<history-conditional>
```

Optional extra check that already exists — adaptation vs the 17:44 incumbent, not vs always-FIGHT / always-ACCOMMODATE:

```text
python market_entry_selfplay.py opponent=frozen train_role=entrant frozen_opponent=type_conditional
```

After both roles can condition on history, a second self-play tests whether reputation survives when both policies move.
