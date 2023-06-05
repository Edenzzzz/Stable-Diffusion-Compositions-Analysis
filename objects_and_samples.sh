input="prompts/one_object_prompts.txt"
while IFS= read -r line
do
  CUDA_VISIBLE_DEVICES=1 python3 scripts/txt2img_demo.py --prompt "$line" --n_samples 4 --n_iter 4 --scheduler dpm --no_grid --outdir "outputs/eval-txt2img-one-object-and"
done < "$input"

input="prompts/two_objects_and_prompts.txt"
while IFS= read -r line
do
  CUDA_VISIBLE_DEVICES=1 python3 scripts/txt2img_demo.py --prompt "$line" --n_samples 4 --n_iter 4 --scheduler dpm --no_grid --outdir "outputs/eval-txt2img-two-objects-and"
done < "$input"

input="prompts/three_objects_and_prompts.txt"
while IFS= read -r line
do
  CUDA_VISIBLE_DEVICES=1 python3 scripts/txt2img_demo.py --prompt "$line" --n_samples 4 --n_iter 4 --scheduler dpm --no_grid --outdir "outputs/eval-txt2img-three-objects-and" 
done < "$input"


input="prompts/four_objects_and_prompts.txt"
while IFS= read -r line
do
  CUDA_VISIBLE_DEVICES=1 python3 scripts/txt2img_demo.py --prompt "$line" --n_samples 4 --n_iter 4 --scheduler dpm --no_grid --outdir "outputs/eval-txt2img-four-objects-and" 
done < "$input"