container_id=$(docker container ls | grep "of:train" | tr -s ' ' | cut -d' ' -f1)
docker exec -it ${container_id} /bin/bash
