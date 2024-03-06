# LBAC
for i in {1..1}
do
	echo "LBAC Run $i"
	python -m main --env-name Drone_3D \
    --LBAC --lambda_LBAC 2000 --logdir drone_xz --logdir_suffix LBAC \
     --num_eps 10000 --constraint_reward_penalty 2000 --method-name LBAC_draw_clb --seed $i\
   --batch_size 512 --alpha 0.2 --warm_start_num 0 --experiment_name Test --lr 0.03
done
