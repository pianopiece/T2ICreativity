images_generated_folder=${1:-"images_generated/model_name"}
images_reference_folder=${2:-"cocoval2014_path"}
images_info=${3:-"benchmark/data.jsonl"}
save_path=${4:-"benchmark/results_model_name.jsonl"}

python eval/eval_creativity.py \
--images-generated-folder ${images_generated_folder} \
--images-reference-folder ${images_reference_folder} \
--images-info ${images_info} \
--save-path ${save_path}