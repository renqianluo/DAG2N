MODEL=ENAS_3_36
echo Using CUDA $1, train model

export  CUDA_VISIBLE_DEVICES=$1
MODEL_DIR=model/model_$MODEL
LOG_DIR=log
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR
nohup python main_es.py \
  --data_dir=data \
  --model_dir=$MODEL_DIR \
  --train_epochs=630 \
  --N=3 \
  --filters=36 \
  --num_nodes=7 \
  --drop_path_keep_prob=0.6 \
  --batch_size=128 \
  --epochs_per_eval=5 \
  --lr_max=0.05 \
  --lr_min=0.001 \
  --T_0=10 \
  --dag='ENAS' \
  --lr_schedule=cosine >$LOG_DIR/train.$MODEL.log 2>&1 &
