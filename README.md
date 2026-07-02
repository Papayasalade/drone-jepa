# drone-jepa

A **sim-only reproduction of SkyJEPA** ([arXiv 2606.23444](https://arxiv.org/abs/2606.23444)) —
a ~9K-parameter JEPA-style latent world model for quadrotors, wrapped in a
sampling-based MPPI controller — plus everything it took to make it *reliably*
fly: a Rust port of the simulator, a training/selection recipe that survives
the weight-init lottery, an RL baseline, and two browser demos.

The paper's central claim, reproduced here: **predicting the next latent
embedding and decoding through a physics-inspired prober tames compounding
error over long horizons, compared to direct autoregressive next-state
prediction** (~8× lower 2-second error vs a naive baseline, on two vehicles).

The harder lesson this repo documents: **whether a trained world model can
actually fly is invisible to every training metric** — two checkpoints with
identical validation error can differ between winning 12/12 races and
crashing on every attempt. The training pipeline here selects on flight, not
loss. See [`docs/EXPERIMENTS_fragility_campaign.md`](docs/EXPERIMENTS_fragility_campaign.md).

## The demos

Everything runs client-side in WASM: the ground-truth simulator, the JEPA's
forward pass, the planners.

- **Race** (`web-demo/web/race.html`) — a perfect-model MPPI, a reactive RL
  policy, and the JEPA-MPPI race the same gate course. All learned brains are
  ~9.5K params and control raw per-rotor forces.
- **Follow** (`web-demo/web/forecast.html`) — a drone flies an aerobatic path
  while the JEPA forecasts its next half-second (the green ghost). Gusts,
  payload drops, airframe swaps, and a speed knob that pushes it out of its
  training envelope.

The two model blobs the demos embed (`bigquad_unified.jblob`, the JEPA, and
`skyrl_rotor_bigquad.rlb`, the RL policy — ~120 KB total) are committed, so the
demos build out of the box. Other checkpoints are not committed; train your own
via the Quickstart (~80 min) and export with `scripts/export_jepa_blob.py`.

```bash
# build the WASM bundle once, then serve statically
scripts/build_wasm.sh     # wasm-pack with source paths remapped
cd web-demo/web && python3 -m http.server 8000
# -> http://localhost:8000/race.html and /forecast.html
```

## Layout

```
drone_jepa/               # the model + training (PyTorch)
  model/
    encoders.py           # state TCN [8,8,16] -> s(16); action TCN [4,4,8] -> z(8)
    predictor.py          # single-layer GRU, hidden 24 (latent dynamics)
    prober.py             # 3-layer MLP -> DKI residuals (see --prober-inputs)
    dki.py                # differentiable kinematic integrator (SO(3) exp), 0 params
    sigreg.py             # SIGReg anti-collapse regularizer
    jepa.py               # assembled model (+ deployment predict())
  train.py                # two-stage training (--warm-start, --ema-tau, ablations)
  train_recipe.py         # THE recipe: N seeds -> probe -> flight gate -> winner
  eval/
    openloop.py           # compounding-error benchmark (--probe adds the honest metric)
    probe.py              # deployment-action plan probe (predicts flyability)
  control/mppi.py         # Python MPPI (reference implementation)
  rl/                     # PPO baseline (PufferLib on the Rust env)

web-demo/
  racer/                  # app layer on the rotor-rs simulator (a crates.io
                          # dependency: crates.io/crates/rotor-rs, sources at
                          # github.com/Papayasalade/rotor-rs): SkyJEPA inference, MPPI
                          # planners, gates, RL runner, WASM demo bindings,
                          # model assets, and all data-gen/benchmark examples
    examples/             # rotor_fly (race benchmark), gen_dataset*, dagger_collect, ...
  rl-env/                 # vectorized C-ABI RL environment (~2.5M steps/s)
  jepa-rs/                # zero-dep JEPA forward pass (native + WASM), golden-tested
  web/                    # the two demos (three.js + the wasm bundle)

scripts/                  # core pipeline: convert/export/select/probe
scripts/experiments/      # one-off campaign drivers (research record)
docs/                     # blog post, experiment logs, design notes
drones/                   # drone-spec JSONs (vehicle definitions)
```

## Quickstart: train a flying model from scratch

Requires Python 3.11+ (`pip install -e .`) and a Rust toolchain.

```bash
# 1. build the simulator + harnesses
cd web-demo/racer && cargo build --release --examples && cd ../..
# (the rotor-rs simulator is fetched from crates.io automatically)

# 2. generate training data (8000 trajectories, ~5 min, all cores)
web-demo/racer/target/release/examples/gen_dataset_rf 8000 200 artifacts/data.bin 7
python scripts/convert_dataset.py artifacts/data.bin artifacts/data

# 3. the recipe: train 4 seeds, probe-rank, confirm by racing, keep the winner
python -m drone_jepa.train_recipe --data artifacts/data.pt --stem mymodel \
    --n-seeds 4 --device mps
# (~40 min. Do NOT skip the selection: seeds with identical val loss range
#  from 12/12 wins to crashing every race. train_recipe.py's docstring is the
#  distilled recipe, including what provably does not work.)

# 4. optional but worth it — one DAgger round (collect the planner's own
#    mistakes, retrain warm-started; took our models from 8/12 to 12/12):
ROTOR_BLOB=assets/mymodel.jblob web-demo/racer/target/release/examples/dagger_collect \
    artifacts/dagger.bin 2500 0
python scripts/dagger_mix.py artifacts/data.pt artifacts/dagger.bin artifacts/data_dagger.pt
python -m drone_jepa.train --data artifacts/data_dagger.pt --warm-start artifacts/mymodel.pt \
    --action-mode rotor_force --pos-mode relative --device mps --out artifacts/mymodel_v2.pt

# 5. benchmark + deploy
python -m drone_jepa.eval.openloop --jepa artifacts/mymodel_v2.pt \
    --baseline artifacts/baseline.pt --data artifacts/data.pt --probe
python scripts/export_jepa.py artifacts/mymodel_v2.pt mymodel_v2 && \
    python scripts/export_jepa_blob.py mymodel_v2
```

Or the whole select→eval→deploy flow as one command: `python scripts/pipeline.py`.
A non-default vehicle is one JSON file (`drones/bigquad.json`) passed as `--drone`.

## Reproducing the RL baseline

The reactive PPO racer (~9.5K params, same budget as the JEPA) trains in about
6 minutes on a laptop — the vectorized Rust environment steps 4,096
domain-randomized drones at ~2.5M transitions/s through a C ABI.

```bash
pip install -e ".[rl]"                       # adds pufferlib
cd web-demo/rl-env && cargo build --release && cd ../..   # the env cdylib

# train the rotor-force policy on the bigquad family (the committed one):
RL_ACTION_MODE=rotor MASS_LO=0.9 MASS_HI=1.5 SYM=1 ARM_LO=0.2 ARM_HI=0.3 \
    python -m drone_jepa.rl.train_puffer --steps 60000000 --envs 4096 \
    --out artifacts/skyrl_rotor.pt
# (omit RL_ACTION_MODE for CTBR through the inner rate loop; omit the MASS/ARM
#  band for the full 0.2-2 kg asymmetric fleet — the regime where RL shines
#  and the world model does not, see docs/EXPERIMENTS_jepa_fleet.md)

# export for the demo / native harnesses, then validate on the 12-race benchmark
python scripts/export_rl_blob.py artifacts/skyrl_rotor.pt skyrl_rotor
cd web-demo/racer && RL_ROTOR=1 RL_BLOB=assets/skyrl_rotor.rlb \
    DRONE_MASS=1.2 DRONE_IXX=0.0189 DRONE_IYY=0.0191 DRONE_IZZ=0.0365 \
    DRONE_ARM=0.25 DRONE_K_ETA=3.92e-6 DRONE_K_M=9.6e-8 DRONE_TAU_M=0.02 \
    ./target/release/examples/rl_fly
# reference result for the committed policy: 10/12 wins, 0 crashes, 4.6 gates/race
```

Environment details (obs/action/reward) are documented at the top of
`web-demo/rl-env/src/lib.rs`; the policy network in `drone_jepa/rl/policy_net.py`.

## Results (bigquad, 1.2 kg / thrust-to-weight 3, everything from scratch)

| | open-loop pos err @1 s | @2 s | 12-race benchmark |
|---|---|---|---|
| JEPA (9,947 params, after DAgger) | 0.59 m | 1.80 m | **12/12 wins, 0 crashes, 0 respawns** |
| naive autoregressive baseline | — | ~14 m | — |
| strong multi-step AR baseline | 0.63 m | 1.50 m | — |
| RL policy (9,524 params, PPO) | n/a (reactive) | n/a | 10/12 wins, 0 crashes |

Notable negative results, all documented with experiments: validation loss
does not predict flight; EMA target encoders make everything worse; principled
init schemes don't fix the seed lottery; early stopping can't detect it; the
paper-literal latent-only prober diverges (ours reads latent + running state +
action — the one architectural deviation, ablated in `NOTES.md`).

## Docs

- [`NOTES.md`](NOTES.md) — every spec-gap decision, deviation, and diagnosis.
- [`docs/EXPERIMENTS_fragility_campaign.md`](docs/EXPERIMENTS_fragility_campaign.md) —
  the nine-experiment investigation of why training is a lottery and what to do.
- [`docs/EXPERIMENTS_jepa_fleet.md`](docs/EXPERIMENTS_jepa_fleet.md) — the
  wide-fleet study (rank-fidelity, why RL tolerates width and JEPA-MPPI doesn't).
Most of this code was written by Claude (Anthropic's model), directed and
reviewed by a human.

## License

MIT — see [LICENSE](LICENSE).
