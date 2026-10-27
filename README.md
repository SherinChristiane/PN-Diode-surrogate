# pn-diode-surrogate

Neural net surrogate for a silicon PN diode, trained on ~2000 DEVSIM simulations. Takes NA, ND, lifetime and length, and gives the forward I-V curve (0.05–0.8 V). One prediction takes ~0.1 ms, versus ~1.7 s for the simulation.

The network predicts the step-by-step rise in log J instead of the raw values, so the curve is always increasing and the ideality factor stays ≥ 1 (above 0.15 V).

![](results/iv_predictions.png)

On 301 test devices:
- RMSE ~0.02 decades (~1–2% error in J)
- 0 unphysical curves, vs 17% for a plain MLP of the same size

I also used it for inverse design (finding device parameters from a target I-V curve). 19/20 matched when re-checked in DEVSIM. The lightly doped side is recovered much better than the heavily doped side, which makes sense since J0 depends mostly on the lower doping.

### Run

```
pip install -r requirements.txt
python src/generate_dataset.py   # optional, data is already in data/
python src/train.py
python src/inverse_design.py
```

Simple model: constant mobility, no BGN/Auger, 1D, forward bias only.
