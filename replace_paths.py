import json
import os

def replace_val_with_test(json_file_path, output_file_path=None):
    """
    读取JSON文件，将所有路径中的'val/'替换为'test/'，并保存结果
    
    参数:
    json_file_path: 输入JSON文件的路径
    output_file_path: 输出JSON文件的路径，如果为None则覆盖原文件
    """
    if output_file_path is None:
        output_file_path = json_file_path
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 遍历JSON数据并替换路径
        for key, item in data.items():
            if 'direct_path' in item and isinstance(item['direct_path'], str):
                item['direct_path'] = item['direct_path'].replace('val/', 'test/')
            
            if 'inv_path' in item and isinstance(item['inv_path'], str):
                item['inv_path'] = item['inv_path'].replace('val/', 'test/')
            
            if 'image_path' in item and isinstance(item['image_path'], str):
                item['image_path'] = item['image_path'].replace('val/', 'test/')
        
        # 保存修改后的JSON数据
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"成功将路径从'val/'修改为'test/'并保存到 {output_file_path}")
        
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")

# 使用示例
# replace_val_with_test('原始文件.json', '修改后文件.json')
# 如果要覆盖原文件，可以这样调用:
replace_val_with_test('/home/aibocheng/data/MR2/en_test.json')