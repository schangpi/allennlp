DATADIR=./dataset/tagger
TMPDIR=/data/tagger
TASK_NAME=$1
MODEL=$2
CUDA=$3

echo "$TASK_NAME"
echo "$MODEL"
echo "$CUDA"

rm -r "$TMPDIR"/"$TASK_NAME"_"$MODEL"
python -m allennlp.run train json/"$MODEL".json \
--serialization-dir "$TMPDIR"/"$TASK_NAME"_"$MODEL" \
--cuda-device "$CUDA" \
--train "$DATADIR"/train/"$TASK_NAME"_train.txt \
--dev "$DATADIR"/dev/"$TASK_NAME"_dev.txt \
--test "$DATADIR"/test/"$TASK_NAME"_test.txt \
 > "$TMPDIR"/"$TASK_NAME"_"$MODEL"_log