MODEL=AmoebaNet_A_6_36
echo Using CUDA $1, train model $MODEL

export  CUDA_VISIBLE_DEVICES=$1
MODEL_DIR=model/model_$MODEL
LOG_DIR=log
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR
nohup python main_es.py \
  --data_dir=data \
  --model_dir=$MODEL_DIR \
  --train_epochs=600 \
  --N=6 \
  --filters=36 \
  --num_nodes=7 \
  --drop_path_keep_prob=0.7 \
  --batch_size=64 \
  --epochs_per_eval=10 \
  --lr_max=0.024 \
  --lr_min=0.0 \
  --T_0=600 \
  --dag='AmoebaNet_A' \
  --lr_schedule=cosine >$LOG_DIR/train.$MODEL.log 2>&1 &
