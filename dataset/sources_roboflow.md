# Sources for the Roboflow images (`rf_*` files)

The `rf_` images were picked by hand from three public datasets on Roboflow
Universe and collected in one Roboflow project ("Swimmer eval", version 1).
All three are licensed **CC BY 4.0**, which allows reuse with credit.

| Dataset | Author | Images used | What they show | Link |
|---|---|---|---|---|
| Swimmer-detection | Mydata | 60 | Indoor pool, low poolside angles, different distances | https://universe.roboflow.com/mydata-s7f97/swimmer-detection-wafmh |
| drowning detection | Yuki | 15 | Wave pool from a lifeguard-tower angle, plus open water and lake shots | https://universe.roboflow.com/yuki-68xu7/drowning-detection-e6kbk-lxylk |
| water-rescue-dataset | CuiZH | 195 | Mostly people in open water and at sea, some pools, a few empty-water scenes | https://universe.roboflow.com/cuizh/water-rescue-dataset |

**Total: 270 images, 892 swimmer boxes** (relabelled Nov 10, 2026; see below).

## Relabelling (Nov 10, 2026)

The first version of these labels (456 boxes) came straight from the
original datasets, which mostly boxed only the person in trouble. Other
people in the water had no box, and some boxes covered only a head, so
YOLO got marked wrong for finding real people.

All 270 images were relabelled:

- Boxes were drawn automatically with Meta's SAM 3 model in Roboflow
  (prompt: "person"), a different model from the YOLO26n being tested, so
  the test stays fair.
- Every image was then reviewed by hand: boxes on people out of the water
  (pool deck, beach, boats) or on non-people were removed, and missed
  swimmers were added.
- Rule used: box every person in the water you can tell is a person,
  including people in the background. People out of the water are not boxed.
- Result: 892 swimmer boxes; 9 images have no people in the water.

Roboflow project: "Swimmer eval relabel", version 1.

## Earlier changes to the original labels

- The original datasets used the classes `swimmer`, `swimming` and `drowning`.
  All were merged into one class, `swimmer` (class id 0), to match the rest
  of this dataset.
- The water-rescue-dataset also boxed floating objects (`float`). Those 23
  boxes were deleted, because they aren't people. The images were kept, so
  they now test whether YOLO raises false alarms on floating objects.
- Near-duplicates, augmented copies (grayscale, rotated, sheared) and
  directly overhead or drone shots were left out.

## Known limitations

- Some images are stock photos of staged drowning, so they're closer to the
  camera than a boat would usually be.
- Roboflow's filenames are long and include the original upload name; the
  image source is the dataset above, not a specific photographer.
