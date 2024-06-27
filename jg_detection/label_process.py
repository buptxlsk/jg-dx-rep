import os

def modify_first_number_to_zero(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # 修改第一行的第一个数字
            if lines:
                first_line = lines[0].split()
                if first_line:
                    first_line[0] = '0'
                    lines[0] = ' '.join(first_line) + '\n'
            
            # 写回修改后的内容
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(lines)
            
            print(f'修改文件: {file_path}')

# 示例用法
folder_path = '/home/cxl/xwz/jg_detection/data/labels'  # txt文件夹路径
modify_first_number_to_zero(folder_path)
