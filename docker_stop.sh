container_id=$(docker container ls | grep "of:train" | tr -s ' ' | cut -d' ' -f1)
docker kill ${container_id}
