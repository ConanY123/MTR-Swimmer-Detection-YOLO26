# Training data for the fine-tuned swimmer model

The fine-tuned model (`models/yolo26n_swimmer.pt`) is trained on images that
are **separate from the eval set** in `dataset/images`. The training set isn't
stored in the repo: `scripts/build_train_set.py` downloads and builds it
(see `notebooks/train_swimmer_colab.ipynb`).

| Dataset | Author | Version | What it adds | License | Link |
|---|---|---|---|---|---|
| tinyperson | Chris D | 3 | Far-away people at sea and on beaches | CC BY 4.0 | https://universe.roboflow.com/chris-d-dbyby/tinyperson |
| Swimmer detection 2 | test | 2 | Lap swimmers lying flat in pools | CC BY 4.0 | https://universe.roboflow.com/test-p2znd/swimmer-detection-2 |

Neither was used for the eval set (that came from Mydata, Yuki and CuiZH,
see `sources_roboflow.md`). They were picked because the eval showed YOLO26n
misses mostly small, far-away swimmers and swimmers lying flat in the water.

## What the build script changes

- **TinyPerson:** only people in the water are boxed (same rule as the eval
  set). People on the beach are painted over with the photo's average
  colour, so the model isn't taught that they're background (version 2). Each photo is cut into 4 tiles so the
  tiny people are bigger relative to the image. About 80% of tiles with
  nobody in the water are skipped.
- **Swimmer detection 2:** every 3rd image is kept (neighbouring video frames
  are near-copies), underwater shots are dropped, and the 640x640 squash
  Roboflow applied is undone by stretching back to 16:9.
- **Overlap check:** any training image that is a near-copy of an eval image
  is removed.
- Each dataset's own train/valid split is kept: train images are used for
  training, valid/test images to decide when to stop. The eval set is only
  used at the end, to compare the normal and fine-tuned model.

## Known limitations

- The 16:9 un-squash is a guess; a few pool images may not be 16:9 originally.
- No close-up drowning photos are in the training set, so the fine-tuned
  model should be checked on large, close swimmers too (the eval set's
  "large" group), not only on small ones.

## Results

| Model | Recall (conf 0.25) | Precision | Notes |
|---|---|---|---|
| yolo26n.pt (normal) | 23.8% | 72.6% | |
| yolo26n_swimmer (v1, Nov 18) | 4.9% | 52.9% | Much worse. Results in `eval_results/per_image_yolo26n_swimmer_conf_*.csv` |

Version 1 did well on its own validation images (precision 0.89, recall
0.82) but forgot large, close people: it found 0 of the 14 people in the
open-water photos (the normal model finds 13). Two likely causes: the
training set has almost no medium or large people, and TinyPerson's beach
people were left unboxed, teaching the model that ordinary-looking people
are background. Version 2 paints over the beach people, keeps all 80 COCO
classes so the original person detector is kept, locks the first 10 layers
and uses a lower learning rate (`scripts/train_swimmer.py`).
