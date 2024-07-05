import os

def rename_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 过滤出图片文件，可以根据需要修改文件扩展名
    image_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    # 按文件名排序
    image_files.sort()
    
    # 临时重命名以避免冲突
    temp_names = []
    for index, file in enumerate(image_files):
        # 获取文件扩展名
        file_extension = os.path.splitext(file)[1]
        
        # 构造临时文件名
        temp_name = f"temp_{index}{file_extension}"
        temp_names.append(temp_name)
        
        # 构造完整的文件路径
        old_path = os.path.join(folder_path, file)
        temp_path = os.path.join(folder_path, temp_name)
        
        # 重命名文件为临时名称
        os.rename(old_path, temp_path)
        print(f"Renamed {file} to {temp_name}")
    
    # 重新获取排序的临时文件名
    temp_names.sort()
    
    # 最终重命名为目标名称
    for index, temp_name in enumerate(temp_names, start=1):
        # 获取文件扩展名
        file_extension = os.path.splitext(temp_name)[1]
        
        # 构造新的文件名
        new_name = f"{index}{file_extension}"
        
        # 构造完整的文件路径
        temp_path = os.path.join(folder_path, temp_name)
        new_path = os.path.join(folder_path, new_name)
        
        # 重命名文件为目标名称
        os.rename(temp_path, new_path)
        print(f"Renamed {temp_name} to {new_name}")

# 使用前请将 'your_folder_path' 替换为实际的文件夹路径
your_folder_path = "D:\Code\jg_dx_yolo\my_dataset\output_crops"
rename_images(your_folder_path)
