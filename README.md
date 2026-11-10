# space-awareness-ml

Two studies on public data, by Samiur Rahman Khan, York University.

Both are about space situational awareness: noticing an object in orbit, and then deciding what it is. The first uses pictures from a star tracker. The second uses the way a rocket body brightens and dims as it tumbles. The questions are small on purpose, and the answers below are the measurements, including the places where an idea did not help.

The longer write-up is the page in `docs/index.html`. Every number on that page, and every number in the tables here, is read from the result files. Nothing in the tables was typed in by hand.

## The two questions

A star tracker spends its life matching stars to a catalogue so the spacecraft can know which way it is pointing. Other things cross the same pictures: working satellites, spent rocket bodies, and debris. In one frame, those resident space objects look like stars. They are a few pixels wide, and many of them are as bright as the stars beside them.

The DUST paper measured that likeness directly. Classifiers given a 15 by 15 pixel patch reach a ROC-AUC of only 0.72 to 0.75: better than a coin toss, and not good enough to trust. The paper's own conclusion is that motion across frames is what separates an object from a star. OrbitTrack follows that conclusion by stacking two raw frames and letting the network learn the motion.

There is a catch on this particular spacecraft. FAI is spin-stabilised, so the stars move from frame to frame as well. A network shown two raw frames has to learn the camera's motion and the object's motion at the same time. The star tracker has already done the first of those. Matching the stars is how it solves its attitude. The first study asks how much a detector gains when that plate solution is used to line the previous frame up before the two frames are compared.

The second study starts from a different operational fact. A light-curve classifier is trained on a fixed list of rocket-body types. The next object an operator sees is often not on that list. A 2026 scoping review (Hwang, Suthakar, Qashoa, Lee, and Sohn) sets out the gap between the labels in public datasets and what operations actually need. Two measurements follow, on one public benchmark. How long does the brightness curve have to be watched before the type can be named? And if the object belongs to a type the model was never trained on, can the model say so?

## What came out of it

**Registration is the gain in the pictures.** The same small detector, trained the same way, scores an F1 of 0.013 ± 0.022 on a single frame, 0.678 ± 0.035 when it is shown the raw previous frame as well, and 0.824 ± 0.020 when that previous frame is aligned on the stars and subtracted. F1 here is the balance of two rates: precision, the share of detections that were real, and recall, the share of real objects that were found. A single frame barely finds these objects. Handing the network the camera motion, already solved, is worth about 0.146 in F1 over asking it to learn that motion itself.

**Telling the detector where the catalogue stars are does not add a further gain.** The residual plus a star mask scores 0.822 ± 0.032. The gap to the residual alone is 0.002, smaller than the scatter between training seeds. In the bin the mask was meant to help, objects within 5 pixels of a catalogue star, recall with the mask is 0.078 lower than recall without it.

**The plate solution has to be present.** On the session of 20 June 2023, 2 of 38 frames were aligned from plate-solved stars. There the raw pair recalls 0.528 ± 0.035 and the aligned residual recalls 0.089 ± 0.019. Subtraction hurts when the stars were not actually lined up.

**Two minutes of a rocket-body light curve carries most of the closed-set answer.** Accuracy is 0.543 ± 0.005 at 30 seconds, 0.703 ± 0.009 at 120 seconds, and 0.778 ± 0.015 at both 600 and 1,000 seconds. The curve stops rising at 600 seconds partly because the records themselves run out: the median test curve has its last real sample at 577 seconds, and 53% of test curves end before 600 seconds.

**No one score can flag every unseen rocket-body type.** A nearest-neighbour score on the network's embedding has the highest average, an AUROC of 0.756. AUROC is the chance that a true unknown is ranked as more unknown than a known object. A half is a coin toss. That average hides the spread: 0.944 on Ariane 5, 0.558 on Falcon 9. Scores read from the classifier and scores read from the embedding fail on different types, which is the finding that points at the next experiment.

## Detection: what was compared, and why

The pictures are the DUST benchmark, 256 by 256 frames from the FAI star tracker, released with the Scientific Data paper. One set has manual labels and the other has semi-automatic labels. Training follows the split in OrbitTrack's released code, because that is the comparison worth making. The model trains on the semi-automatic set, holds out two whole sessions for validation (`2023_05_10` and `2023_07_20`), and is tested on all of the manual set: 866 frames and 2,168 objects. The first frame of every session is dropped, since it has no previous frame.

Frames are not shuffled into train and test. Neighbouring frames are nearly the same picture, and a random split would let the model memorise a view it is about to be tested on. That is also why the EfficientDet number reported in the DUST paper, from a random frame split, sits in the table as context rather than as a peer of these rows.

The detector is YOLO11n in every run. Only the three input channels change. The same network, the same split, and the same three seeds mean that a change in the score is a change in the picture.

| Input | What the network sees | Precision | Recall | F1 | Average precision |
|---|---|---:|---:|---:|---:|
| V0, single frame | the current frame, repeated | 0.072 ± 0.124 | 0.007 ± 0.012 | 0.013 ± 0.022 | 0.093 ± 0.005 |
| V1, raw pair | previous frame and current frame, unaligned | 0.650 ± 0.087 | 0.715 ± 0.029 | 0.678 ± 0.035 | 0.707 ± 0.013 |
| V2, registered residual | current frame, and the previous frame after the stars are lined up and subtracted | 0.796 ± 0.036 | 0.854 ± 0.007 | 0.824 ± 0.020 | 0.843 ± 0.013 |
| V3, residual and star mask | the same residual, plus a channel marking catalogue stars | 0.802 ± 0.070 | 0.846 ± 0.015 | 0.822 ± 0.032 | 0.837 ± 0.006 |

The ± is the sample standard deviation over three seeds. A difference inside that scatter is not a result. The operating point is a confidence of 0.25. Average precision uses the whole range of confidences. Box mAP50, on boxes grown to at least 6 by 6 pixels, is a secondary number: 0.085 ± 0.007, 0.706 ± 0.012, 0.828 ± 0.006, and 0.830 ± 0.013 for V0 through V3. The objects are 1 to 3 pixels across, and overlap of boxes that small is a poor score, so the primary test is whether the predicted centre falls within 3 pixels of a labelled centre. Each label can be claimed once, and claims are taken in order of confidence.

Alignment uses the star matches the tracker already computed. Stars are paired from frame to frame by catalogue right ascension and declination. The per-frame index number is not stable across frames, so it is not used as an identity. When at least four stars agree, a similarity transform is fit with RANSAC. Of the prepared frames, 1,028 (75.8%) were aligned this way, with a median error of 0.36 pixels. The other 328 (24.2%) fall back to an intensity alignment (ECC) on a high-pass image, so a few very bright stars do not dominate the fit. The residual keeps the positive part of the difference: a pixel lights up where something is present now and was not present in the lined-up previous frame.

The star mask is a filled disk, radius 2 pixels, at each catalogue star whose match weight is at least 0.5. The hope was practical. If the detector is told which pixels are known stars, it should raise fewer false alarms beside them, and it might do better on faint objects that a raw stack confuses with the star field. Two checks were built in before that hope was trusted. About two thirds of labelled objects already sit at the brightest value the sensor can record (66%), so brightness is the sum of a 5 by 5 window with the local background removed, not the peak pixel. And 88 of 4,183 labelled object positions fall inside a mask disk (2.1%), so the mask rarely covers an object by accident.

The breakdowns say where the registration actually helps. Among brightness and speed, the largest gain of the residual over the raw pair is the slow third, objects moving under 3.6 pixels per frame: recall 0.353 ± 0.039 for the raw pair and 0.734 ± 0.011 for the residual, a gap of 0.380. A slow object is the one most easily mistaken for a star that is only drifting because the spacecraft is spinning. Once the stars cancel, it is what remains. The dimmest third improves as well, from 0.437 ± 0.021 to 0.597 ± 0.023, a smaller gap of 0.160. Fast objects are already found by the raw pair (recall 0.969 ± 0.009), and the residual does not move that number.

The mask does not deliver the help it was drawn for. Within 5 pixels of a catalogue star, recall is 0.867 ± 0.003 without the mask and 0.789 ± 0.025 with it. Far from stars the two are level. The headline F1 values were already inside each other's seed scatter. The close-star bin is stronger than that: there the mask costs recall.

The session of 20 June 2023 is the honest failure, and it belongs in the result. With 2 of 38 frames aligned from plate-solved stars, recall is 0.528 ± 0.035 for the raw pair, 0.089 ± 0.019 for the residual, and 0.139 ± 0.111 for the residual plus the mask. A subtraction of two frames that were not really aligned erases objects and invents leftovers. Any onboard use of this idea needs a check that the plate solution succeeded, and a path back to the raw pair when it did not. That switch was not built here. The measurement says it would matter.

A constant-velocity tracker was run on one trained model, V3 with seed 0, at the same confidence of 0.25. It predicts the next position from the last step, accepts a detection within 6 pixels (15 pixels on the first link of a track), drops a track after two missed frames, and keeps a track with at least three points. On the manual set this gives MOTA 0.768 and IDF1 0.836, with 16 identity switches across 78 objects, 65 false detections, and 421 misses. MOTA counts misses, false detections, and swaps together. IDF1 asks whether each object kept one identity for the whole pass. This is one run. It shows that these detections can be strung into tracks. It is not a tracker comparison.

Two published numbers are context, from different pipelines. OrbitTrack reports precision 0.972 and recall 0.742 on DUST. The DUST paper reports an EfficientDet at mAP50 0.61, precision 0.73, and recall 0.51, on a random frame split. They are useful as a sense of the field. They are not rows of this table.

## Light curves: what was compared, and why

The curves are RoBo6, brightness records of six rocket-body types from the Mini-MegaTORTORA survey, released as a NeurIPS 2024 workshop benchmark. They are already on a 10 Hz grid. A missing sample is stored as zero, and a second channel tells the network which samples were really observed, so a gap is never read as a brightness of zero. Each curve is standardised using its observed samples only. The released train and test files are used as published. Those files contain 128 exact duplicate rows in training and 27 in the test set. The duplicates were kept, so the counts stay the ones in the paper. The test set has 1,404 curves.

The network is a small one-dimensional ResNet. It pools only the timesteps that were observed. During training, half the time a random stretch between 30 and 1,000 seconds is kept and the rest is hidden. The model therefore practises on short looks, which is the situation the budget test later measures. That choice is one reason a comparison with the paper's own training is context rather than a paired result: the split is theirs, the training is not.

Closed set, all six types, the full 1,000 seconds, three seeds:

| | Accuracy | Macro-F1 |
|---|---:|---:|
| This ResNet | 0.778 ± 0.015 | 0.772 ± 0.011 |
| RoBo6 paper, ResNet | 0.694 | 0.600 |
| RoBo6 paper, Astroconformer | 0.725 | 0.684 |

The three seeds landed at 0.793, 0.764, and 0.776. The paper rows are the authors' numbers on this split with their own code. Landing above them says the pipeline is healthy and the training choices are competitive with the published figures. It does not isolate which choice produced the difference.

The budget test shows each test curve cut to its first N seconds. Accuracy, mean ± standard deviation over the three seeds:

| Seconds | 30 | 60 | 120 | 300 | 600 | 1000 |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.543 ± 0.005 | 0.641 ± 0.006 | 0.703 ± 0.009 | 0.750 ± 0.014 | 0.778 ± 0.010 | 0.778 ± 0.015 |

At two minutes the accuracy is 0.703, against 0.778 when the model is allowed the whole record. Most of what this classifier can do with a full curve, it can already do with two minutes. Nothing is added between 10 minutes and the full 1,000 seconds: both read 0.778.

That plateau needs the coverage number beside it. The median test curve has its last observed sample at 577.2 seconds. Of the 1,404 test curves, 52.9% have ended before 600 seconds, 47.1% have any sample after 600 seconds, and 15.5% have any sample after 900 seconds. Past about 10 minutes, many of the "longer" inputs are the same pixels as the shorter ones, with zeros after the telescope stopped. Part of the flat line is the data running out. A claim that the classifier has saturated would need curves that truly continue, and this test set does not supply them for most objects. The paper's ResNet, given the full curve and their own code, is reported at 0.694. Reading the two-minute result against that figure mixes our training with their number, so it stays context.

For unknown detection, one type is removed from training and the model is trained on the other five. On the test set, the held-out type should be scored as unknown, and the five known types should still be classified. This is repeated for every type and for three seeds. No unknown examples are shown during training. Five scores are computed afterwards:

- Max softmax, max logit, and energy read how sure the classifier is of its own answer. A hesitant answer is treated as a sign that the object may be new.
- Mahalanobis distance and a 10-neighbour cosine distance read the 128-dimensional embedding, the network's internal summary of the curve. An object far from every training example is treated as new.

Mean AUROC over the six held-out types: max softmax 0.665, max logit 0.670, energy 0.671, Mahalanobis 0.616, nearest neighbour 0.756. Nearest neighbour is the best average, and the average is the wrong place to stop. Its AUROC by held-out type is Ariane 5 at 0.944 ± 0.014, H-2A at 0.910 ± 0.018, Atlas 5 Centaur at 0.784 ± 0.024, Delta 4 at 0.699 ± 0.005, CZ-3B at 0.643 ± 0.027, and Falcon 9 at 0.558 ± 0.019. Falcon 9 is close to a coin toss. The stricter operating point makes the same point more sharply. FPR95 is the fraction of unknown objects still accepted when 95% of known objects are accepted. For nearest neighbour that fraction is 0.961 ± 0.015 on Falcon 9 and 0.957 ± 0.025 on Delta 4. A score can look useful on average and still wave through the type an operator would most want caught.

The two families also disagree with each other, which a single average would hide. On Delta 4, energy reaches 0.885 and Mahalanobis falls to 0.337, below chance, a lead of 0.549 for energy. On H-2A the lead flips: Mahalanobis reaches 0.847 and energy falls to 0.481, a lead of 0.366 the other way. Classifier-output scores and embedding scores fail on different types. On the budget figure, the spread drawn around the unknown-detection curve is over six held-out types times three seeds. It mostly records that some types are harder to flag than others.

One reading note for that figure: the known-class accuracy during leave-one-out is a five-class accuracy. It is a different quantity from the six-class closed-set accuracy, and the two lines are not one curve with a dip in it.

## What was deliberately left out

The studies answer the questions above. Several neighbouring questions were in view and were not run. They are listed here so the boundary of the result stays visible.

The star mask was one design: disks of radius 2 at match weight 0.5 or above, offered as an extra channel. A softer mask, a different radius, or a training loss that refuses to fire inside the mask was not tried. The present result is already informative about where to spend that effort. On the headline the mask sits inside the seed scatter, and in the close-star bin it lowers recall. Redrawing the same channel is a weak next step. The open problem the data does point to is the frames the plate solution cannot align. A rule that trusts the residual only when the star fit succeeded, and uses the raw pair otherwise, was not built or measured. The 20 June session says that rule has something to recover.

Tracking is one motion model on one seed. There is no sweep of the gate, no other motion model, and no comparison against OrbitTrack's tracker. MOTA and IDF1 are a check that the detections form tracks.

The unknown scores are all post hoc. Nothing was trained with an "unknown" label, and no score adds the classifier output to the embedding distance. The heatmap is the reason to try that combination: the two families fail on different types, so a sum, a product, or a small model on top of both has a chance of covering more of the table than either family alone. That measurement is the obvious one left on the table. It has not been run.

The budget test always keeps the start of the curve. It does not ask whether a later window of the same length would do as well, and it does not ask whether an observer could stop early once a score was already confident. Both are closer to how a telescope is actually scheduled. Both need the coverage caveat kept beside them, because a "later window" on a curve that ended at 577 seconds is often empty.

Three items from the wider plan are not started, and are future work rather than missing pieces of these two results:

- An event-camera spiking detector (EBSSA), which would change the sensor rather than the input channels.
- Pretraining on simulated star fields with SatSim, then testing the move to the real DUST frames.
- An onboard integer (INT8) version of the detector, which is a deployment question and was not part of the measurement.

Nothing here claims a camera other than FAI, or a survey other than Mini-MegaTORTORA. A gain from the plate solution on this star tracker is evidence about this star tracker.

## The page

`docs/index.html` is a single column that fetches `a_results.json`, `b_results.json`, `coverage.json`, and a small export of frames. GitHub Pages should serve the `/docs` folder. The sky viewer shows two whole sessions, 31 March 2023 and 20 June 2023, with the labelled objects and the seed-0 detections drawn on the frames. The confidence slider filters those stored detections. It does not retrain anything, and the trails are the published V3 seed-0 tracks at confidence 0.25, so they stay put while the slider moves. The frames are from DUST, which is CC BY 4.0.

## Reproduce

Open `colab/run_all.ipynb` in Google Colab on a T4, set `REPO_URL` in the config cell, and run all. Finished stages skip themselves, and checkpoints are written to Google Drive so a disconnect does not require a full rerun. The last cell zips the results.

The detection tables are the first completed training of all 12 runs (four inputs, three seeds). A later session retrained six of those runs after a disconnect. Those later weights are not the numbers in `results/a_results.json`. Scoring them again would be a separate reproducibility check, and it should be written to a different file. Re-running the original scoring cell would overwrite the record this README describes.

## Sources and licences

- DUST: Suthakar, Kunalakantha, Lee, and Sohn, Scientific Data, 2026, [doi:10.1038/s41597-026-07736-9](https://doi.org/10.1038/s41597-026-07736-9). Data: [Zenodo 10.5281/zenodo.20255672](https://doi.org/10.5281/zenodo.20255672), CC BY 4.0. The 15 by 15 patch result (ROC-AUC 0.72 to 0.75) is quoted from that paper.
- OrbitTrack: Jeong, Suthakar, Qashoa, Sohn, and Lee, Advances in Space Research, 2025, [doi:10.1016/j.asr.2025.08.006](https://doi.org/10.1016/j.asr.2025.08.006). Precision 0.972 and recall 0.742 are from their paper, on a different pipeline.
- RoBo6: Kyselica, Šuppa, Šilha, and Ďurikovič, NeurIPS 2024 ML4PS workshop, [arXiv:2412.00544](https://arxiv.org/abs/2412.00544). Dataset [`kyselica/RoBo6`](https://huggingface.co/datasets/kyselica/RoBo6), MIT. The ResNet and Astroconformer rows are their reported numbers, same split, their implementation.
- Scoping review: Hwang, Suthakar, Qashoa, Lee, and Sohn, Aerospace, 2026, [doi:10.3390/aerospace13030287](https://doi.org/10.3390/aerospace13030287).
- Detector training uses ultralytics, AGPL-3.0. The training channels are not colours, so the default colour jitter is turned off.

Repository: [github.com/samiurk70/space-awareness-ml](https://github.com/samiurk70/space-awareness-ml).
