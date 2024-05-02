python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py  -c config/ESTS/ESTS_5scale_multi_finetune.py --coco_path /data/hmx/video_data/image_data --output_dir logs_cross/croos_domain_prompt_attn_TT_CTW \
        --train_dataset totaltext_train:ctw1500_train \
        --val_dataset totaltext_val \
        --pretrain_model_path checkpoint.pth \
        --find_unused_params \
        --options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0