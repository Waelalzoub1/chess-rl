# chess-rl

AlphaZero-style chess. A residual policy + value network guides a PUCT Monte Carlo Tree Search; self-play games are generated on the GPU with batched search; the network is trained from a replay buffer to imitate the search's visit distribution and to predict game outcomes. A web viewer shows the current checkpoint playing itself.

Measured strength after 60 iterations (13,440 self-play games, 101 minutes on one RTX 5080), MCTS with 200 simulations per move, half the games as white:

| opponent | games | W / D / L | score | 95 % Wilson interval |
|---|---|---|---|---|
| random mover | 400 | 381 / 19 / 0 | **0.976** | 0.956 – 0.987 |
| random mover (separate 100-game match) | 100 | 91 / 8 / 1 | 0.950 | 0.888 – 0.978 |
| greedy-capture baseline | 100 | 42 / 56 / 2 | 0.700 | 0.604 – 0.781 |

The agent never trained against either opponent. It reliably beats random play; against an opponent that simply grabs material it rarely loses but converts fewer than half its games, mostly because winning positions end as draws by repetition or at the 300-ply cap.

## How it works

- **Search** (`chessrl/batched.py`): PUCT, score = Q + c·P·√(ΣN + 1)/(1 + N) with c = 1.5. Many independent game trees advance in lockstep, so each simulation step is one network forward pass over all games. Terminal positions are scored by the rules: −1 for the side that is mated, 0 for stalemate, insufficient material and the 50-move rule. A leaf whose position already occurred earlier in the game or on the current search path is scored as a draw, so the search does not shuffle into repetitions when it thinks it is ahead. Dirichlet noise (α = 0.3, 25 %) at the root during self-play.
- **Network** (`chessrl/model.py`): conv stem, 6 residual blocks of 96 channels, a policy head (1×1 conv → linear to 4096) and a value head (1×1 conv → MLP → tanh). 3,131,449 parameters. The size is stored in the checkpoint.
- **State** (`chessrl/encode.py`): 17 × 8 × 8 binary planes from the side-to-move's perspective (6 own-piece planes, 6 opponent, 4 castling rights, 1 en passant). The board is mirrored when black is to move, so the network only ever sees "my pieces move up the board".
- **Actions**: 4096 = from-square × to-square. Underpromotions are folded into the queen promotion.
- **Reward**: +1 / −1 for checkmate, 0 for a draw (stalemate, insufficient material, 50-move rule, threefold repetition). Games still running at 160 plies are truncated with a shaped reward, clip(material balance × 0.05, ±0.5), so that long undecided games still carry a signal; `--no-shaping` turns that off.
- **Training** (`train.py`): 7 actor processes share the GPU, each playing 32 concurrent games with 64 simulations per move against the latest checkpoint. Positions go into a 400,000-position replay buffer; each iteration the learner takes 300 AdamW steps (batch 1024, lr 1e-3) on policy cross-entropy + value MSE, then writes a new checkpoint. Every fifth iteration a 40-game match against the random mover is logged.
- **Baselines** (`chessrl/arena.py`): `random` moves uniformly at random; `greedy` captures the most valuable piece it can (ties at random) and otherwise moves at random. Matches are played concurrently with batched search; a game is a draw at threefold repetition, the 50-move rule, stalemate, insufficient material, or after 300 plies.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m unittest discover -s tests -v                       # encoding, value sign, mate in one, repetition
python evaluate.py --games 100 --sims 200                     # committed checkpoint vs random and greedy
python train.py --iters 60 --seed 0 --out-dir runs/s0 --fresh # retrain from scratch (needs a CUDA GPU to be practical)
python watch.py --port 8000                                   # live board at http://127.0.0.1:8000
```

## Training run

`python train.py --iters 60 --actors 7 --games-per-actor 32 --concurrent 32 --sims 64 --train-steps 300 --seed 0` on one RTX 5080 (16 GB) and 8 CPU cores: 224 games and about 31,000 positions per iteration, 74 s of self-play per iteration on average, 1.88 M positions in total. `checkpoints/stats.json` has the per-iteration log.

![learning curves](docs/learning_curves.png)

Score against the random mover during training (40 games, 64 simulations): 0.75 at iteration 5, 0.81 at 10, 0.85 at 15, 0.86 at 20, then between 0.89 and 0.93 from iteration 25 to 60, with no losses after iteration 10. With 200 simulations the same checkpoints score higher (iteration 20: 35 / 5 / 0 in 40 games).

Self-play outcomes are not a strength measure: both sides are the same network, so wins split evenly and many games stay undecided however strong it gets. For the record, after iteration 5 about 25 % of self-play games end in checkmate, 6 % in a draw, and 69 % are truncated at 160 plies. Strength is measured only against the fixed baselines above. The 56 draws against the greedy-capture baseline, a far weaker opponent, are the evidence for the agent's main weakness: it wins material but is slow to convert it into mate.

## Tests

`tests/test_chessrl.py`: every legal move round-trips through its policy index; a position and its colour-mirrored twin encode identically and mirrored moves get the same index; the search finds mate in one (for white and for black) with an untrained network; the search value is positive for the side that can mate and negative for the side that cannot avoid it; a repeated position is scored as a draw while a mating move from the same position is not.

## History

The first version of this project ran self-play on CPU with one position per network call, trained only on the latest iteration's games, and did not see repetitions in search. Measured the same way, it did not beat a random mover: 0 wins, 18 draws, 2 losses in 20 games after 63 iterations, and 2 wins in 480 games across six seeded 12-iteration runs. The changes that made the difference were batched GPU self-play (about 3.5× more games per hour, which paid for a larger network and more simulations), the replay buffer, and the repetition-aware search.

## Known limitations

- Weak at converting won positions; no endgame knowledge beyond what the search sees.
- Underpromotions are not representable in the policy.
- Truncation with material shaping is a heuristic that biases the value head toward material.
- One training run with one seed is reported; the variance between seeds has not been measured for this version.
- python-chess move generation on the CPU is the throughput limit, not the GPU.

## Layout

```
chessrl/encode.py    board and move encoding
chessrl/model.py     policy + value network
chessrl/batched.py   batched, repetition-aware PUCT search
chessrl/mcts.py      single-position wrapper around the batched search
chessrl/selfplay.py  concurrent self-play, actor entry point
chessrl/arena.py     baselines and matches
train.py             actors, replay buffer, learner
evaluate.py          strength against the baselines
watch.py             live self-play viewer
tests/               unit tests
checkpoints/         model.pt (iteration 60) and stats.json
```

---

<sub>Wael Alzoubi · [mini-os32](https://github.com/Waelalzoub1/mini-os32) · [custom-16bit-CPU](https://github.com/Waelalzoub1/custom-16bit-CPU) · [Transformer-character-level](https://github.com/Waelalzoub1/Transformer-character-level)</sub>
