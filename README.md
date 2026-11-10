# space-awareness-ml

Two small experiments on public space situational awareness data.

- **A. Star-prior registered differencing for RSO detection on DUST.** Use the star tracker's own plate solution to register consecutive frames, subtract, and add a catalogue star mask, then measure how much detection recall improves over single-frame and raw two-frame inputs.
- **B. Open-set recognition and observation budget on RoBo6 rocket-body light curves.**

Work in progress. Results tables will be filled from `results/*.json` once the runs finish.

## Reproduce

Open `colab/run_all.ipynb` in Google Colab with a T4 GPU, set `REPO_URL` in the config cell, and run all.

## Data

- DUST, Zenodo record 20255672, CC BY 4.0.
- RoBo6, Hugging Face `kyselica/RoBo6`, MIT licence.
