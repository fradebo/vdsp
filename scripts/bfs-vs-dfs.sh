#!/bin/sh

cd ../trees/bfs-vs-dfs-trees
rm -rf *
cd ../..
python3 scripts/bfs-vs-dfs.py
wait
cd First\ Passage
for file in ../trees/bfs-vs-dfs-trees/*; do 
    if [ -f "$file" ]; then
        echo $(basename ${file})
        cp "$file" inp.json
        cp "../proc_order/$(basename ${file})" proc_order.json
        ./tree 
        wait
        python3 get_F.py "$file" "../evaluation/bfs-vs-dfs-bfs-11.csv"
    fi 
done