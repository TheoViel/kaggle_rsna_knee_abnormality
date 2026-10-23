#!/bin/bash
# Reproduces rsna-knee-weights-1 from the competition data, run from src/:
#
#   bash reproduce.sh            # every stage, in order
#   bash reproduce.sh roi crops  # some stages (each needs the outputs of the previous ones)
#
# Stages: data, side, cls, roi, crops, teachers, students, package. Each run logs to ../logs/repro/<name>/<date>/<id>/;
# a later stage takes the latest run of that name. Needs input/ (train.csv, train_series.csv, train_series/), the report
# labels in data/labels/ and 4 GPUs (NGPU to change). See README.md for what each stage does.
set -e
cd "$(dirname "$0")"
NGPU=${NGPU:-4}
GPUS=$(seq -s, 0 $((NGPU - 1)))
PY=${PY:-python}
T="${TORCHRUN:-torchrun} --standalone --nproc_per_node=$NGPU"
LOGS=${LOGS:-../logs/repro}
OUT=${OUT:-../output}
run() { ls -d "$LOGS/$1"/*/*/ | sort -V | tail -1; }    # latest run folder of a named run

data() {
    $PY prepare_data.py --workers 64                     # DICOM -> output/series_npy, metadata, series_index.csv
    $PY prepare_cache.py --workers 96                    # -> output/cache_384 (uint8, 384 px, <= 64 slices)
    $PY -m roi.prepare                                   # sagittal / coronal -> 32 x 128 x 128 volumes
    $PY -m roi.prepare --planes Axial --name volumes_axial
}

side() {
    $PY side_cls.py tags                                 # labels: the DICOM Laterality tag
    $T main_side.py --log_path $LOGS/side/
    $PY side_cls.py predict --run "$(run side)"          # the model's side of every series, used from here on
}

cls() {
    # The 12-finding classifier with attention, 4 variants whose attention maps are ensembled.
    $T main_cls.py --log_path $LOGS/cls_224/
    $T main_cls.py --log_path $LOGS/cls_f10/ --n_frames 10 --val_n_frames 10
    $T main_cls.py --log_path $LOGS/cls_288/ --img_size 288
    $T main_cls.py --log_path $LOGS/cls_small/ --backbone convnext_small.fb_in22k
}

roi() {
    local runs
    runs="$(run cls_224),$(run cls_f10),$(run cls_288),$(run cls_small)"
    $PY -m roi.extract --tag ens --runs "$runs" --studies all --gpus "$GPUS"
    # Round 0: compact attention boxes that the other view does not contradict (no Baker's yet).
    $PY -m roi.labels attention --tag ens --out round0 --drop_classes "Baker's"
    $T main_roi.py --labels round0 --epochs 40 --log_path $LOGS/roi_round0/
    # Round 1: U-Net boxes confirmed by another series of the study replace / add labels.
    $T main_roi.py --predict "$(run roi_round0)" --pred_out round0_pred
    $PY -m roi.labels merge --prev round0 --pred round0_pred --out round1
    $T main_roi.py --labels round1 --epochs 20 --log_path $LOGS/roi_round1/
    # Round 2: + MCL from an anatomical prior (the attention never finds the ligament) and Baker's from attention.
    $T main_roi.py --predict "$(run roi_round1)" --pred_out round1_pred
    $PY -m roi.mcl_prior --pred round1_pred --out mcl_prior
    $PY -m roi.labels bakers --tag ens --pred round1_pred --out bakers0
    $PY -m roi.labels merge --prev round1 --pred round1_pred --extra mcl_prior,bakers0 --out round2
    $T main_roi.py --labels round2 --epochs 20 --log_path $LOGS/roi_round2/
    # Round 3: MCL from the prior only, re-placed on the round-2 compartments.
    $T main_roi.py --predict "$(run roi_round2)" --pred_out round2_pred
    $PY -m roi.mcl_prior --pred round2_pred --out mcl_prior
    $PY -m roi.labels merge --prev round2 --pred round2_pred --extra mcl_prior --drop_classes MCL --out round3
    $T main_roi.py --labels round3 --epochs 20 --log_path $LOGS/roi_round3/           # the weights-1 U-Net
    $T main_roi.py --predict "$(run roi_round3)" --pred_out round3_pred
    $PY -m roi.evaluate round0_pred round1_pred round2_pred round3_pred
}

crops() {
    $PY -m crops.prepare --boxes round3_pred --name r3
    $PY -m crops.axial_boxes --pred round3_pred --out round3_axial
    $PY -m crops.prepare --boxes round3_axial --name r3_axial --volumes volumes_axial
}

# Crop models trained on the report labels, scored on the holdout; the 5 best per finding are the teachers.
# name: main_crops.py arguments (the original run number first).
declare -A TEACHERS=(
    [t06]="--group ligaments" [t07]="--group acl" [t08]="--group mcl" [t09]="--group bakers" [t10]="--group all"
    [t15]="--group menisci --img_size 224"
    [t19]="--group menisci --backbone maxvit_tiny_tf_224.in1k --img_size 224 --batch_size 16"
    [t20]="--group menisci --backbone coatnet_0_rw_224.sw_in1k --img_size 224 --batch_size 16"
    [t22]="--group oa --img_size 224" [t23]="--group oa --head mean"
    [t26]="--group oa --backbone maxvit_tiny_tf_224.in1k --img_size 224 --batch_size 16"
    [t27]="--group oa --backbone coatnet_0_rw_224.sw_in1k --img_size 224 --batch_size 16"
    [t28]="--group oa --epochs 15"
    [t30]="--group oa --crops r3,r3_axial --planes Sagittal,Coronal,Axial"
    [t31]="--group bakers --crops r3,r3_axial --planes Axial"
    [t32]="--group bakers --crops r3,r3_axial --planes Sagittal,Coronal,Axial"
    [t34]="--group mcl --crops r3,r3_axial --planes Sagittal,Coronal,Axial"
    [t39]="--group acl --labels mean" [t40]="--group mcl --labels mean" [t41]="--group menisci --labels mean"
    [t42]="--group oa --labels mean" [t43]="--group bakers --labels mean"
    [t44]="--group acl --labels codex" [t45]="--group mcl --labels codex" [t46]="--group menisci --labels codex"
    [t47]="--group oa --labels codex" [t48]="--group bakers --labels codex"
)
declare -A TEACHERS_OF=(          # finding -> its 5 teachers (the pseudo-label is their mean)
    ["ACL"]="t39 t06 t44 t10 t07" ["MCL"]="t34 t10 t08 t45 t40"
    ["Medial Meniscus"]="t20 t15 t41 t19 t10" ["Lateral Meniscus"]="t10 t15 t46 t19 t41"
    ["Medial OA"]="t10 t42 t27 t47 t30" ["Lateral OA"]="t22 t26 t30 t27 t28"
    ["PF OA"]="t30 t42 t23 t47 t22" ["Baker's"]="t43 t32 t48 t09 t31"
)

teachers() {
    local name dirs json=""
    for name in $(printf "%s\n" "${!TEACHERS[@]}" | sort); do
        # shellcheck disable=SC2086
        $T main_crops.py --log_path $LOGS/$name/ ${TEACHERS[$name]}
    done
    dirs=$(for name in $(printf "%s\n" "${!TEACHERS[@]}" | sort); do run "$name"; done | paste -sd,)
    $PY -m crops.predict --runs "$dirs" --gpus "$GPUS"   # every study, -> <run>/all_study_preds.csv
    for c in "${!TEACHERS_OF[@]}"; do
        json+="\"$c\": [$(for name in ${TEACHERS_OF[$c]}; do printf '"%s",' "$(run "$name")"; done | sed 's/,$//')],"
    done
    echo "{${json%,}}" > $OUT/crops/teachers.json
    $PY -m crops.pseudo_labels --teachers $OUT/crops/teachers.json   # -> pl_round1.csv, pl_round2.csv
}

# The submission models: fullfits (every study, the 58 radiologist studies with their own labels) on pl_round2.
declare -A STUDENTS=(
    [acl]="--group acl" [mcl]="--group mcl --crops r3,r3_axial --planes Sagittal,Coronal,Axial"
    [menisci]="--group menisci --backbone coatnet_0_rw_224.sw_in1k --img_size 224 --batch_size 16"
    [all]="--group all" [oa224]="--group oa --img_size 224"
    [oa_axial]="--group oa --crops r3,r3_axial --planes Sagittal,Coronal,Axial" [bakers]="--group bakers"
)

students() {
    local name
    for name in "${!STUDENTS[@]}"; do
        # shellcheck disable=SC2086
        $T main_crops.py --log_path $LOGS/s_$name/ --labels pl:pl_round2 --fullfit True ${STUDENTS[$name]}
    done
}

package() {
    local out=$OUT/weights_1 name
    rm -rf $out && mkdir -p $out/crops
    cp -r "$(run side)" $out/side
    cp -r "$(run roi_round3)" $out/roi
    for name in "${!STUDENTS[@]}"; do
        mkdir -p $out/crops/$name
        cp "$(run "s_$name")"/{args.json,ckpt_final.pt} $out/crops/$name/
    done
    cat > $out/recipe.json <<'EOF'
{"ACL": ["acl", "all"], "MCL": ["mcl", "all"], "Medial Meniscus": ["menisci", "all"],
 "Lateral Meniscus": ["menisci", "all"], "Medial OA": ["all", "oa224", "oa_axial"], "Lateral OA": ["oa224"],
 "PF OA": ["oa_axial"], "Baker's": ["all", "bakers"]}
EOF
    echo "-> $out"
}

for stage in "${@:-data side cls roi crops teachers students package}"; do
    for s in $stage; do echo "=== $s" && $s; done
done
