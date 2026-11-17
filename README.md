# Grip Area Detection for Vacuum Grippers on Narrow Surface Objects

Code of the bachelor thesis "Hochpraezise Greifpunkt-Lokalisierungstechniken fuer Vakuumgreifer in industriellen
Anwendungen" (Turkish-German University, Department of Mechanical Engineering, 2026; advisor: Ali Omer Baykar) and of
the paper "Grip Area Detection for Vacuum Grippers on Narrow Surface Objects" (SCA'26, 11th International Conference
on Smart Cities and Applications, Karabuk University, November 2026; to appear in Springer Lecture Notes in Networks
and Systems).

Authors of the paper: Ozgur Deniz Karagoz, Ali Omer Baykar, Ayhan Kural, Jens Lambrecht, Erdem Onur Ozyurt, Ahmet Ugur Batuk.

This work was supported by the Scientific and Technological Research Council of Turkiye (TUBITAK) under the 2209-A
University Students Research Projects Support Program 2025/1 (project no. 1919B012514934).

The work is split into two problems, as in the thesis:

* **Problem 1 - grip area:** find the flat, narrow strip of an elongated plastic part where a suction cup can seal,
  from a single 2D image.
* **Problem 2 - grip point:** choose the point inside that area where the cup is placed.

| | Method | Folder |
|---|---|---|
| Problem 1 | Method 1: MAT-GOF, masked adaptive thresholding with geometric overlap filtering | `problem1_grip_area/method1_mat_gof` |
| | Method 2: CIELAB-AMF, L-channel enhancement with anisotropic morphological filtering | `problem1_grip_area/method2_cielab_amf` |
| | Method 3: HM R-CNN, classical part cropping + Mask R-CNN on a 160 x 1600 canvas | `problem1_grip_area/method3_hm_rcnn` |
| | Top/bottom side classifier (1D signal along the part + LightGBM) | `problem1_grip_area/orientation_classifier` |
| | YOLOv8s-seg full-frame baseline (paper only) | `problem1_grip_area/yolov8seg_baseline` |
| Problem 2 | Method A: center of the maximum inscribed circle (distance transform) | `problem2_grip_point/method_a_mic` |
| | Method B: spatial error heatmaps + active-learning model committee (thesis only) | `problem2_grip_point/method_b_error_heatmaps` |

The paper evaluates Problem 1 with a geometric suction feasibility rate: a point counts as feasible if
`d >= D/2 + t` (d = distance to the nearest edge of the labeled grip area, D = cup diameter, t = positioning tolerance).

## Layout

```
dataset/                              dataset scans, annotated photos per setup, Figure 2 of the paper
problem1_grip_area/
  method1_mat_gof/                    method_1_sidechecker_debugged_v3.py + its parameter search
  method2_cielab_amf/                 method_2_gripareafinder_v4.py
  method3_hm_rcnn/                    crop generator, Mask R-CNN training, HM R-CNN inference
  orientation_classifier/             signal datasets (precise_cropper.py, datasetcreater_...py) and LightGBM training (trainer.py)
  yolov8seg_baseline/                 COCO -> YOLO conversion, training, mAP, mIoU, latency
problem2_grip_point/
  method_a_mic/                       check that the distance-transform MIC equals the Voronoi MIC
  method_b_error_heatmaps/
    heatmap1_maskrcnn_distance/       1  distance transform of the Mask R-CNN grip mask
    heatmap2_camera_reprojection/     2  per-point reprojection error of the camera calibration (adaptive grid)
    heatmap3_transformation_residual/ 3  residual error of the pixel -> robot transformation, calibration sheet
    heatmap4to6_robot_kinematics/     4-6 lever error, condition number and manipulability of the UR3e
      workspace/                          workspace map with the camera ROI, PyBullet reachability scan
    fusion_active_learning/           fusion of the heatmaps, Gaussian process exploration + weighted model committee
evaluation/
  suction_feasibility/                evaluation chain of the paper (run in this order):
    1_build_test_list.py                labeled validation/test photographs
    2_run_tests.py                      runs the four methods and measures every suction point
    2b_fix_cielab_points.py             re-measures CIELAB-AMF points that fell into padded crop regions
    3_make_excel.py                     Excel workbook of all measurements
    4_table2_from_csv.py                Table 2 and the numbers of the results section
    5_draw_test_results.py              one image per test with its measurement
    additional_measurements/            latency, mIoU on the same parts, rates per photo set, statistics
  early_measurements/                 earlier mIoU and latency measurements (thesis stage)
```

Only the last version of each script is included, plus a few older ones that do something the last version does not:
`fusion_active_learning/older_versions/unifier_v5.py` has the 21-model committee described in the thesis
(`unifier_v6.py` explores with a Gaussian process for the first 40 rounds and then uses 8 models), and the scripts in
`heatmap4to6_robot_kinematics/workspace/` draw the robot workspace and check reachability. The scripts were developed in
three stages called unifier_1, unifier_2 and unifier_3; the data folders under `codes_results/` still use these names.
Heatmaps 7 (radial) and 8 (random control) are read from .pkl files by `unifier_v6.py`.

## Setup

```bash
conda create -n grip python=3.9
conda activate grip
# install PyTorch for your CUDA version from https://pytorch.org first
pip install -r requirements.txt
```

## Data and models

Datasets, trained weights and generated results are not in this repository.

* Scripts look for data under the folder given by the environment variable `GRIP_PROJECT_ROOT`
  (default: `./project_root`). It should contain `datasets/` (e.g. `datasets/derivative_datasets/...`) and,
  for the method B scripts, `codes_results/`.
* The trained HM R-CNN weights go to `models/sametkalip_maskrcnn_160x1600_2_master.pth` (repository root).
* The YOLOv8s-seg script writes its converted dataset, runs and weights next to itself.
* The evaluation scripts write to a `results/` folder next to them.
* `evaluation/early_measurements/time_and_iou_tester_with_cropper_mic.py` downloads its data from Roboflow and reads
  the API key from the environment variable `ROBOFLOW_API_KEY`.

```bash
set GRIP_PROJECT_ROOT=D:\path\to\project_root        # Windows (cmd)
export GRIP_PROJECT_ROOT=/path/to/project_root        # Linux / macOS
```

## Notes

* Some data keys and file names are Turkish because existing CSV, Excel, pickle and model files use them:
  CSV columns such as `foto_no`, `foto_yolu` (photo path), `kaynak` (source), `veri_seti` (dataset),
  `etiket_dosyasi` (label file), `genislik` / `yukseklik` (width / height); the class names `alt` (bottom),
  `griparea`, `ust` (top); the committee model names (`LGBM_Hizli` = fast, `_Derin` = deep, `_Dengeli` = balanced,
  `RF_Klasik`, `NN_Sinir`); columns `X_Piksel`, `Y_Piksel`, `Gercek_Puan` (measured score); and output files such as
  `grasp_success_rate_parcalar.csv` (per part), `test_listesi.csv` (test list), `piksel_hata_haritasi.pkl`
  (pixel error map), `hafiza.csv` (memory) and `rapor.xlsx` (report).
* Most scripts were written for Windows paths (`r"datasets\..."`); on Linux replace the backslashes.
* The evaluation in the paper is geometric: no physical grasp was executed. Method B was not validated on the robot.
