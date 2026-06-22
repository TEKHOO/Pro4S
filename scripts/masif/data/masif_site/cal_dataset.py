import os

path = './lists/test.txt'
error = './error.txt'
error_ = []

with open(path, 'r') as file:
    commands = file.readlines()

# 获取总命令数
total_commands = len([cmd.strip() for cmd in commands if cmd.strip()])
current_count = 0

# 遍历每一行命令并执行
for command in commands:
    # 去除行尾的换行符
    command = command.strip()
    # 只有在行非空的情况下才执行
    if command:
        current_count += 1
        try:
            # 显示进度
            print(f'Masif surface calculate {current_count}/{total_commands}')
            # 执行命令
            os.system('./data_prepare_have_pdb_no_masif.sh ' + command)
        except:
            error_.append(command + '\n')
            print(f'{command} 出错啦')

with open(error, 'w') as f:
    f.writelines(error_)

