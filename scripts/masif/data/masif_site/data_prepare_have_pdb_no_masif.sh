#!/bin/bash
masif_root=../../../masif
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_matlab

if [ "$1" == "--file" ]; then
    echo "Running masif site on $2"
    PPI_PAIR_ID=$3
    PDB_ID=$(echo $PPI_PAIR_ID | cut -d"_" -f1)
    CHAIN1=$(echo $PPI_PAIR_ID | cut -d"_" -f2)
    CHAIN2=$(echo $PPI_PAIR_ID | cut -d"_" -f3)
    FILENAME=$2
    mkdir -p data_preparation/00-raw_pdbs/
    cp $FILENAME data_preparation/00-raw_pdbs/$PDB_ID.pdb
else
    PPI_PAIR_ID=$1
    PDB_ID=$(echo $PPI_PAIR_ID | cut -d"_" -f1)
    CHAIN1=$(echo $PPI_PAIR_ID | cut -d"_" -f2)
    CHAIN2=$(echo $PPI_PAIR_ID | cut -d"_" -f3)
    # 不再需要下载PDB文件的部分，直接使用已有的PDB文件路径
    # 这里假设 PDB 文件已经存在于某个目录（例如 data_preparation/00-raw_pdbs/），
    # 如果需要，你可以根据实际的存储路径调整
    echo "Skipping PDB download, using local PDB file."
fi

# 提取和三角化
if [ -z $CHAIN2 ]; then
    echo "Processing single chain ($CHAIN1)"
    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
else
    echo "Processing both chains ($CHAIN1 and $CHAIN2)"
    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN2
fi

## 预计算
# python $masif_source/data_preparation/04-masif_precompute.py masif_site $PPI_PAIR_ID
