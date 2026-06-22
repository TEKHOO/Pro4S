import os
import time
import shutil

path = './masif/data/masif_site/lists/test.txt'
error = './masif_error.txt'
out_path = './masif/data/masif_site/data_preparation/01-benchmark_surfaces'
surface_path = '../test/surface'
os.makedirs(surface_path, exist_ok=True)
error_ = []

# with open(path, 'r') as file:
#     commands = file.readlines()
# 
# # 遍历每一行命令并执行
# for command in commands:
#     # 去除行尾的换行符
#     command = command.strip()
#     # 只有在行非空的情况下才执行
#     if command:
#         try:
#             # 执行命令
#             os.system('./masif/data/masif_site/data_prepare_have_pdb_no_masif.sh ' + command)
#         except:
#             error_.append(command + '\n')
#             print(f'{command} 出错啦')
# 
# with open(error, 'w') as f:
#     f.writelines(error_)

time.sleep(2)


def clear_and_copy_files_simple(source_dir, target_dir):
    # 清空目标文件夹
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)

    # 复制文件
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            target_path = os.path.join(target_dir, item)

            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)
        print("操作完成！")
    else:
        print("源文件夹不存在")


clear_and_copy_files_simple(out_path, surface_path)

# remove tmp file
if os.path.exists(out_path):
    shutil.rmtree(out_path)
    shutil.rmtree('./masif/data/masif_site/data_preparation/01-benchmark_pdbs')
