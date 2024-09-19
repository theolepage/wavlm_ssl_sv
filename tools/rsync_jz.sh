source_path="."
target_path="jeanzay:~/wavlm_ssl_sv"

rsync -azh $source_path $target_path \
    --progress \
    --force \
    --delete \
    --exclude="slurm_*" \
    --exclude="data" \
    --exclude="exp" \
    --keep-dirlinks

while inotifywait -r -e modify,create,delete $source_path
do
    rsync -azh $source_path $target_path \
          --progress \
          --force \
          --delete \
          --exclude="slurm_*" \
          --exclude="data" \
          --exclude="exp" \
          --keep-dirlinks
done
