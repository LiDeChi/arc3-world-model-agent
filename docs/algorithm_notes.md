# Algorithm Notes

## Why a world model?

ARC-AGI-3 tasks are interactive. A good player must infer that action A now may unlock action B later. The implemented `DreamerLiteWorldModel` gives the agent a small internal simulator:

1. `encode(grid)` compresses a 16-color grid into a latent vector.
2. `predict_next_latent(latent, action)` predicts the next hidden state.
3. `decode(latent)` reconstructs the imagined grid for visualization.
4. `predict_reward(latent, action)` estimates whether the transition is useful.

The production target is Dreamer/MuZero-style RSSM + actor-critic. The v1 code uses a NumPy version so the whole stack runs locally without a GPU.

## Planning rule

For each legal first action, the agent samples several short imagined action sequences. It sums predicted rewards with discounting and subtracts an action penalty. The action with the best average imagined score is selected. This makes action efficiency visible and aligns with RHAE-style scoring.

## Self-tuning rule

The PBT controller trains a population of trials, ranks them by evaluation return and success rate, keeps elites, copies elite configs into weak trials, and perturbs learning rate, epsilon, action penalty, imagination horizon, and planning rollouts. All decisions are logged to `pbt_events.jsonl`.
