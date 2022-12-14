python ../main.py \
       --experiment adept_l2norm_pretrain \
       --dataset atis \
       --name pretrain_adept_sample_experiment \
       --epochs 500 \
       --max_seq_len 20 \
       --batch_size 64 \
       --hidden_size 128 \
       --learning_rate 0.01 \
       --clipping_constant 5 \
       --weight_decay 0 \
       --train_teacher_forcing_ratio 0.0 \
       --optim_type adam \
       --embed_type none \
       --embed_size 300 \
       --seed 42 \
       --early_stopping True \
       --patience 20 \
       --local False \
       --output_dir "<path>/<to>/<output_dir>" \
       --asset_dir "<path>/<to>/<asset_dir>" 
