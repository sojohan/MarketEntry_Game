# Market Entry Self-Play: Code Explanation

This script is essentially a **repeated Schelling-style market-entry game turned into a reinforcement-learning environment for Tinker**. Its research question is whether an LLM can discover three behaviours through repeated play:

1. **Reputation building**
2. **Signaling**
3. **Bayesian-style adaptation**

The easiest way to understand it is to separate the code into four layers:

```text
Game theory
    ↓
MarketEntryCoordinator
    ↓
Tinker environments
    ↓
RL dataset + advantage calculation
    ↓
Model training
```

## 1. The game being simulated

There are two players:

```text
ENTRANT
INCUMBENT
```

The entrant decides:

```text
STAY_OUT
ENTER
```

If the entrant enters, the incumbent decides:

```text
ACCOMMODATE
FIGHT
```

The incumbent has a hidden type:

```text
STRONG
WEAK
```

The type is sampled once at the beginning of the game and stays fixed for all rounds. The incumbent knows its type; the entrant does not.

The payoff matrix is:

| Entrant | Incumbent | Entrant payoff | Strong incumbent | Weak incumbent |
|---|---|---:|---:|---:|
| STAY_OUT | — | 0 | 5 | 5 |
| ENTER | ACCOMMODATE | 3 | 2 | 2 |
| ENTER | FIGHT | -2 | 3 | -2 |

### Strong incumbent

For a strong incumbent:

\[
FIGHT = 3
\]

versus

\[
ACCOMMODATE = 2
\]

So fighting is individually rational.

### Weak incumbent

For a weak incumbent:

\[
FIGHT = -2
\]

versus

\[
ACCOMMODATE = 2
\]

So, **in a one-shot game**, a weak incumbent should accommodate.

But the game is repeated for, by default:

```python
horizon = 10
```

That creates the strategic issue.

A weak incumbent might deliberately accept:

\[
-2
\]

today by fighting, because the entrant may infer:

> "This incumbent is probably STRONG."

and stay out in future rounds.

Then the weak incumbent receives:

\[
+5
\]

in those future rounds.

That is **reputation building**.

---

## 2. Why the hidden type matters

At episode creation:

```python
self.incumbent_type = (
    "STRONG" if rng.random() < p_strong else "WEAK"
)
```

The default probability is:

```python
p_strong = 0.30
```

so:

\[
P(S)=0.30
\]

and:

\[
P(W)=0.70
\]

The entrant sees this prior probability, but does not see the actual type.

Therefore, initially it has a belief:

\[
P(S)=0.3
\]

After observing behaviour, it can implicitly update that belief.

For example:

```text
Round 1:
Entrant ENTER
Incumbent FIGHT
```

The entrant might reason:

\[
P(S\mid FIGHT) > P(S)
\]

Then in round 2 it may become less willing to enter.

This is what the script calls **Bayesian-style adaptation**.

It is not explicitly coding Bayes' theorem. Instead, the experiment tests whether the LLM learns behaviour equivalent to Bayesian updating from history.

---

## 3. `MarketEntryCoordinator` is the game engine

This class contains the actual state of one game:

```python
class MarketEntryCoordinator:
```

It stores things like:

```python
self.incumbent_type
self.round_idx
self.current_player_id
self.pending_entry
self.public_history
self.cumulative_rewards
```

Think of it as the referee.

The LLMs do not decide the payoffs. They only choose actions.

The coordinator enforces the rules.

---

## 4. Each round starts with the entrant

Initially:

```python
self.current_player_id = ENTRANT
```

Then the entrant chooses either:

```text
STAY_OUT
```

or:

```text
ENTER
```

If it chooses `STAY_OUT`, the round immediately ends:

```python
entrant_reward = 0
incumbent_reward = 5
```

So:

```text
Entrant
   │
   ├── STAY_OUT
   │       ↓
   │     (0, 5)
   │
   └── ENTER
           ↓
       Incumbent
```

If it chooses `ENTER`, control switches to the incumbent:

```python
self.pending_entry = True
self.current_player_id = INCUMBENT
```

---

## 5. Then the incumbent responds

After entry:

```text
             ENTER
               │
         ┌─────┴─────┐
         ▼           ▼
   ACCOMMODATE     FIGHT
```

If it accommodates:

```python
entrant = +3
incumbent = +2
```

If it fights:

```python
entrant = -2
```

and the incumbent reward depends upon its private type:

```python
3 if STRONG
-2 if WEAK
```

So the strong and weak incumbent have genuinely different short-term incentives.

That is what makes signaling possible.

---

## 6. Public history creates reputation

At the end of every round the code records:

```python
{
    "round": ...,
    "entrant": ...,
    "incumbent": ...,
    "entrant_payoff": ...,
    "incumbent_payoff": ...
}
```

For example:

```text
Round 1: entrant ENTER, incumbent FIGHT.
Payoffs entrant=-2, incumbent=-2.

Round 2: entrant STAY_OUT.
Payoffs entrant=0, incumbent=+5.

Round 3: entrant STAY_OUT.
Payoffs entrant=0, incumbent=+5.
```

Both agents receive that history in future prompts.

This makes strategies history-dependent:

\[
a_t = \pi(h_t)
\]

where \(h_t\) is the observed history up to time \(t\).

Without that history, there could be no reputation effect.

---

## 7. What the weak incumbent might discover

This is arguably the most interesting part of the experiment.

Suppose there are five rounds remaining.

### Strategy A: always accommodate

If the entrant enters every round:

\[
5 \times 2 = 10
\]

for the incumbent.

### Strategy B: fight once to build reputation

Weak incumbent fights now:

\[
-2
\]

Suppose this convinces the entrant to stay out for the remaining four rounds:

\[
4 \times 5 = 20
\]

Total:

\[
-2 + 20 = 18
\]

So:

\[
18 > 10
\]

The seemingly irrational action:

```text
FIGHT while WEAK
```

may therefore be rational in the repeated game.

That is exactly what the metric:

```text
incumbent/fight_early
vs
incumbent/fight_late
```

is trying to detect.

If weak incumbents fight considerably more often early than late, that is evidence consistent with reputation building.

---

## 8. The prompts are deliberately asymmetric

The entrant gets told:

> You do NOT observe the incumbent's private type.

and sees the payoff possibilities and public history.

The incumbent gets told:

```text
Your private type is: STRONG
```

or:

```text
Your private type is: WEAK
```

and is explicitly reminded:

> The entrant will observe your action in future rounds, but will NOT observe your type.

This is a useful prompt design because it gives the model everything required to reason strategically without telling it:

> "Build a reputation."

The RL training is supposed to discover that behaviour.

---

## 9. Why only very simple outputs are allowed

The models must output exactly things such as:

```text
ENTRANT:[ENTER]
```

or:

```text
INCUMBENT:[FIGHT]
```

The code even pre-fills:

```python
ROLE_PREFILL = {
    ENTRANT: "ENTRANT:[",
    INCUMBENT: "INCUMBENT:[",
}
```

So the model only needs to generate something like:

```text
ENTER]
```

This is important for RL.

You do not really want the model generating a long explanation such as:

> "Considering the incumbent's previous behaviour, I believe the optimal action is ENTER."

You just want the **action**.

It turns the LLM into a stochastic policy:

\[
\pi_\theta(a\mid s)
\]

where:

```text
state  = game history + role + private information
action = ENTER / STAY_OUT / FIGHT / ACCOMMODATE
```

That is much closer to traditional reinforcement learning.

---

## 10. Invalid-output handling

The function:

```python
normalize_action(...)
```

extracts legal actions from generated text.

If the model gives an invalid action, it does not immediately terminate.

It gets up to:

```python
MAX_FORMAT_RETRIES = 2
```

and receives:

```python
FORMAT_PENALTY = -1
```

If it still does not comply, it receives:

```python
INVALID_ACTION_REWARD = -5
```

That is why the script tells you to examine the formatting metrics before interpreting any strategic results.

Ideally:

```text
format/valid        → ~1
format/retry        → ~0
never_acted         → ~0
rounds_completed    → horizon
```

If formatting is poor, apparent "strategy" may simply be malformed generations.

---

## 11. The reward-delta mechanism

This code is easy to miss:

```python
def _reward_delta(self):
    total = self.coordinator.cumulative_rewards[self.player_id]
    delta = total - self.last_seen_reward
    self.last_seen_reward = total
    return delta
```

Why do this?

Because one model can be waiting while the other agent acts.

For example:

```text
Entrant ENTER
       ↓
entrant environment waits
       ↓
Incumbent FIGHT
       ↓
payoffs happen
       ↓
entrant wakes up
```

The entrant's reward happened while its environment was waiting.

So the coordinator maintains cumulative reward:

\[
R_t
\]

and the environment retrieves only the difference:

\[
r_t = R_t-R_{t-1}
\]

This lets each agent correctly recover the reward that accumulated since its previous action.

---

## 12. There are two training modes

The script implements:

```text
1. frozen opponent
2. self-play
```

They are conceptually different experiments.

### Frozen mode

Example:

```bash
python market_entry_selfplay.py \
    opponent=frozen \
    train_role=incumbent \
    frozen_opponent=mixed
```

Here:

```text
Trainable incumbent
        vs
Scripted entrant
```

or the reverse.

The frozen strategies include:

```text
always_enter
always_stay_out
mixed_enter

always_fight
always_accommodate
mixed_respond
type_conditional
```

This mode is easier to learn from because the environment is stationary.

The opponent is not changing while the policy learns.

---

## 13. Why `mixed` is useful

When training the incumbent:

```python
mixed →
    always_enter
    mixed_enter
```

When training the entrant:

```python
mixed →
    always_fight
    always_accommodate
```

Therefore the model cannot simply memorize:

```text
Against this opponent always do X.
```

It sees different strategic conditions and has some chance of learning a more general policy.

---

## 14. Frozen mode uses GRPO-like groups

`FrozenMarketEntryGroupBuilder` creates multiple independent copies:

```python
num_envs = 8
```

Conceptually:

```text
Same trainable incumbent

Game 1 vs scripted entrant
Game 2 vs scripted entrant
Game 3 vs scripted entrant
...
Game 8 vs scripted entrant
```

The different trajectories obtain different returns.

GRPO can then compare:

\[
R_1,R_2,\ldots,R_8
\]

and calculate relative advantages.

This is structurally similar to the Tinker multi-agent tutorial.

---

## 15. Self-play mode

Here one game contains exactly:

```text
Agent 0 = Entrant
Agent 1 = Incumbent
```

Both are live policy trajectories sharing the same coordinator.

Conceptually:

```text
                Qwen policy
                 /       \
                /         \
        Entrant role    Incumbent role
              │             │
              └──── game ───┘
```

These are not necessarily two separate LLMs.

They are two role-conditioned trajectories from the **same underlying trainable policy**.

That means the model learns:

\[
\pi_\theta(a \mid h,\text{role})
\]

rather than having separate parameters:

\[
\pi_{\theta_E}
\]

and

\[
\pi_{\theta_I}
\]

---

## 16. Why normal GRPO would be wrong here

In a normal two-player game, suppose:

```text
Entrant reward   = -2
Incumbent reward = +3
```

It would be tempting to center them against one another.

But that would imply:

```text
incumbent better than entrant
```

which is not meaningful because they are playing **different roles with different payoff scales and decisions**.

So the code changes the advantage calculation.

Instead, it compares entrants against other entrants:

\[
A_E^i =
R_E^i -
\overline{R_E}
\]

and incumbents against other incumbents:

\[
A_I^i =
R_I^i -
\overline{R_I}
\]

This is what the following section does:

```python
entrant_mean = ...
incumbent_mean = ...

return [
    torch.tensor([
        row[0] - entrant_mean,
        row[1] - incumbent_mean
    ])
]
```

Suppose four games produce:

```text
          Entrant    Incumbent

Game 1      3          20
Game 2     -4          27
Game 3      8          14
Game 4      1          22
```

The entrant baseline is:

\[
\bar R_E=2
\]

while the incumbent baseline is:

\[
\bar R_I=20.75
\]

Game 1 therefore gives:

\[
A_E = 3-2=+1
\]

but:

\[
A_I=20-20.75=-0.75
\]

That is much more sensible than comparing entrant and incumbent returns directly.

---

## 17. What the strategy metrics tell you

The code defines the metrics needed for the research question.

### Signaling

```text
fight_if_strong
fight_if_weak
```

You want to examine:

\[
P(FIGHT\mid STRONG)
\]

versus

\[
P(FIGHT\mid WEAK)
\]

If:

\[
P(FIGHT|STRONG)
>
P(FIGHT|WEAK)
\]

then incumbent actions contain information about type.

That creates a signal.

### Reputation building

Compare:

```text
fight_early
fight_late
```

If weak incumbents behave something like:

\[
P(FIGHT|WEAK,\ early)
>
P(FIGHT|WEAK,\ late)
\]

that is especially interesting.

Early fighting has many future rounds in which reputation can pay off.

Late fighting does not.

This is essentially the classic reputation result the experiment is targeting.

### Entrant adaptation

The code also records:

```text
enter_after_fight
enter_after_accommodate
```

You would expect successful learning to produce:

\[
P(ENTER\mid FIGHT)
<
P(ENTER\mid ACCOMMODATE)
\]

because `FIGHT` should increase the entrant's belief that it faces a strong incumbent.

Conceptually:

```text
                   observe FIGHT
P(Strong)=0.30  ─────────────────→  P(Strong | FIGHT) ↑
                                            │
                                            ▼
                                  probability of ENTER ↓
```

That is the Bayesian-style adaptation the script is trying to observe.

---

## 18. The three phenomena reinforce one another

The experiment can produce a feedback loop:

```text
STRONG incumbents FIGHT
        ↓
FIGHT becomes evidence of STRONG
        ↓
Entrants stay out after FIGHT
        ↓
FIGHT develops reputational value
        ↓
WEAK incumbents sometimes imitate STRONG
        ↓
Signal becomes less perfectly informative
        ↓
Entrant must infer probabilities from history
```

That is much richer than simply learning:

```text
ENTER is good/bad.
```

The learned policy potentially has to reason about **beliefs about beliefs**.

The weak incumbent's problem is effectively:

> What will the entrant infer about my hidden type if I choose FIGHT?

And the entrant's problem is:

> Given the action history, what type is this incumbent likely to be?

That is close to the logic of incomplete-information repeated games.

---

## 19. Why frozen mode should probably come first

The script recommends running the same-role GRPO experiment against scripted opponents before live self-play.

That makes methodological sense.

With a frozen opponent:

```text
Policy changes
Opponent stays fixed
```

The learning target is relatively stationary.

With self-play:

```text
Entrant changes
    ↕
Incumbent changes
```

Each agent is learning against an opponent whose strategy is simultaneously moving.

That creates a non-stationary RL environment.

A strange-looking learning curve may therefore reflect:

- genuine strategy adaptation;
- cycling strategies;
- one role learning faster;
- policy collapse;
- changing incentives created by the other role.

---

## 20. One full episode

Suppose the incumbent happens to be:

```text
WEAK
```

and the horizon is 5.

### Round 1

```text
Entrant: ENTER
Incumbent: FIGHT

Rewards:
Entrant = -2
Incumbent = -2
```

The weak incumbent takes a deliberate loss.

### Round 2

The entrant observes the fight:

```text
Entrant: STAY_OUT

Rewards:
Entrant = 0
Incumbent = +5
```

### Round 3

```text
Entrant: STAY_OUT
Incumbent = +5
```

### Round 4

The entrant tests the incumbent:

```text
Entrant: ENTER
Incumbent: FIGHT
```

Another:

\[
-2
\]

for the weak incumbent.

### Round 5

```text
Entrant: STAY_OUT
Incumbent = +5
```

Incumbent cumulative payoff:

\[
-2+5+5-2+5=11
\]

Compare that with accommodating every entry if the entrant consequently kept entering:

\[
5\times2=10
\]

So reputation building can outperform myopic rationality.

---

## 21. The core research experiment

The entire script can be reduced to this:

```text
                Hidden incumbent type
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
            STRONG                 WEAK
              │                     │
              └───────┬─────────────┘
                      │
                   history
                      │
                      ▼
Entrant ── ENTER / STAY_OUT
                      │
                    ENTER
                      ▼
Incumbent ─ FIGHT / ACCOMMODATE
                      │
                      ▼
                   payoff
                      │
                      ▼
               update policy
                      │
                      ▼
                  next round
```

The experiment asks whether RL produces:

\[
\boxed{\text{Signaling}}
\]

\[
\boxed{\text{Reputation}}
\]

\[
\boxed{\text{Belief-dependent adaptation}}
\]

without explicitly coding those strategies into the agents.

The reward function only gives the economic payoffs. The higher-level strategic concepts are supposed to **emerge from repeated interaction**.

---

## Recommended evaluation improvement

One metric should be made more specific before treating the experiment as strong evidence of Schelling-style reputation.

Instead of only tracking:

```text
fight_early
fight_late
```

also track:

```text
fight_if_weak_early
fight_if_weak_late
```

The key comparison is:

\[
P(FIGHT\mid WEAK,\ early)
\quad \text{vs} \quad
P(FIGHT\mid WEAK,\ late)
\]

because the strategically surprising behaviour is specifically a **weak incumbent fighting early despite the immediate -2 payoff**, in order to influence the entrant's future beliefs and actions.
