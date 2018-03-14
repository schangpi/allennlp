TASK_EXT=$1
TMPDIR=$2
CUDA=$3

DATADIR=./dataset/multi_tagger_clean

for ext in "$TASK_EXT"
do
JSONDIR=multi
TASK_NAME=multitagger
MODEL="$JSONDIR"_"$ext"
rm -r "$TMPDIR"/"$TASK_NAME"_"$MODEL"
python -m allennlp.run train json/"$JSONDIR"/"$MODEL".json \
--serialization-dir "$TMPDIR"/"$TASK_NAME"_"$MODEL" \
--cuda-device "$CUDA" \
--train "$DATADIR"/train \
--dev "$DATADIR"/dev \
--test "$DATADIR"/test \
 > "$TMPDIR"/"$TASK_NAME"_"$MODEL"_log

JSONDIR=taskonly_embedding
TASK_NAME=taskembtagger
MODEL="$JSONDIR"_tagger_"$ext"
rm -r "$TMPDIR"/"$TASK_NAME"_"$MODEL"
python -m allennlp.run train json/"$JSONDIR"/"$MODEL".json \
--serialization-dir "$TMPDIR"/"$TASK_NAME"_"$MODEL" \
--cuda-device "$CUDA" \
--train "$DATADIR"/train \
--dev "$DATADIR"/dev \
--test "$DATADIR"/test \
 > "$TMPDIR"/"$TASK_NAME"_"$MODEL"_log

JSONDIR=taskonly_prepend_embedding
TASK_NAME=taskembtagger
MODEL="$JSONDIR"_tagger_"$ext"
rm -r "$TMPDIR"/"$TASK_NAME"_"$MODEL"
python -m allennlp.run train json/"$JSONDIR"/"$MODEL".json \
--serialization-dir "$TMPDIR"/"$TASK_NAME"_"$MODEL" \
--cuda-device "$CUDA" \
--train "$DATADIR"/train \
--dev "$DATADIR"/dev \
--test "$DATADIR"/test \
 > "$TMPDIR"/"$TASK_NAME"_"$MODEL"_log
 
done
