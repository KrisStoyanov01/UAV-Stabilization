# LBAC
for i in {1..1}
do
	echo "LBAC Run $i"
    python -m main --env-name Drone_kris \
      --LBAC --lambda_LBAC 2000 --logdir Drone_conf_2 --logdir_suffix LBAC \
      --num_eps 10000 --constraint_reward_penalty 1000 --method-name LBAC_draw_clb \
    --batch_size 512 --alpha 0.20 --warm_start_num 0 --experiment_name kris2 --lr 0.3 \
    --seed 111111
done

