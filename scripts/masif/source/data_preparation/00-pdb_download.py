#!/usr/bin/python
import Bio
from Bio.PDB import * 
import sys
import importlib
import os
import requests

from default_config.masif_opts import masif_opts
# Local includes
from input_output.protonate import protonate

if len(sys.argv) <= 1:
    print("Usage: "+sys.argv[0]+" PDBID_A_B")
    print("A or B are the chains to include in this pdb.")
    sys.exit(1)

if not os.path.exists(masif_opts['raw_pdb_dir']):
    os.makedirs(masif_opts['raw_pdb_dir'])

if not os.path.exists(masif_opts['tmp_dir']):
    os.mkdir(masif_opts['tmp_dir'])


#
# new
#

in_fields = sys.argv[1].split('_')
pdb_id = in_fields[0]

# Download pdb
file_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

# 使用requests库发送HTTP GET请求来下载文件
response = requests.get(file_url, stream=True)
pdb_filename = os.path.join(masif_opts['tmp_dir'], f"{pdb_id}.pdb")

# 检查请求是否成功
if response.status_code == 200:
    # 打开一个文件句柄，将响应的内容写入文件
    with open(pdb_filename, "wb") as file:
        file.write(response.content)
    print(f"{pdb_id}文件已成功下载到文件夹中。")

else:
    print("请求失败，无法下载文件。")


##### Protonate with reduce, if hydrogens included.
# - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
protonated_file = masif_opts['raw_pdb_dir']+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file


# #
# # 源码
# #
#
# in_fields = sys.argv[1].split('_')
# pdb_id = in_fields[0]
#
# # Download pdb
# pdbl = PDBList(server='http://ftp.wwpdb.org')
# pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masif_opts['tmp_dir'], file_format='pdb')

