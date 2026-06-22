import os
import shutil
import re

def clean_pdb_filenames_strict(folder_path):
    """
    严格清理PDB文件名，只保留字母、数字和文件扩展名

    参数:
    folder_path (str): 存放PDB文件的文件夹路径
    """
    # 只保留字母、数字和文件扩展名中的点
    pattern = r'[^a-zA-Z0-9]'

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        if os.path.isfile(old_path):
            name, ext = os.path.splitext(filename)

            # 去除所有非字母数字字符
            cleaned_name = re.sub(pattern, '', name)

            new_filename = cleaned_name + ext
            new_path = os.path.join(folder_path, new_filename)

            if new_filename != filename:
                try:
                    os.rename(old_path, new_path)
                    print(f"rename: {filename} -> {new_filename}")
                except OSError as e:
                    print(f"rename error {filename}: {e}")

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


pdb_path = '../test/pdb'
clean_pdb_filenames_strict(pdb_path)
txt_masif = './masif/data/masif_site/lists/test.txt'
input_txt = '../test/test.txt'
masif_pdb = './masif/data/masif_site/data_preparation/00-raw_pdbs'

with open(txt_masif, 'w') as f:
    for i in os.listdir(pdb_path):
        name = i.split('.')[0]
        f.write(f"{name}_A_\n")

if not os.path.exists(input_txt):
    with open(input_txt, 'w') as f:
        for i in os.listdir(pdb_path):
            name = i.split('.')[0]
            f.write(f"{name}_0\n")

clear_and_copy_files_simple(pdb_path, masif_pdb)
