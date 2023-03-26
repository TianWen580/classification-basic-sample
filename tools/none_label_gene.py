import os
import json
from tqdm import tqdm

json_sample_file = '23_8.json'
target_path = './'  # ./ 表示对所在目录的图片进行遍历，即标注完成后的图片目录


def change_filename_in_json(params, key, value):
    """
    修改json文件对应关键词的值.

    Parameters:
        params {Any} - json载入的所有结点
        key {str} - 检索json结点的关键词
        value {Any} - 对应关键词位置修改后的值
    Returns:
        修改完成的json结点.
    """
    params[key] = value
    return params


if __name__ == '__main__':
    with open(json_sample_file, 'r') as json_sample:
        params = json.load(json_sample)  # 加载json模版
        progress_bar = tqdm(os.listdir(target_path))
        for filename in progress_bar:
            # 如果图片没有对应标签才生成空标签
            if not os.path.exists(target_path[:-4] + '.json'):
                new_params = change_filename_in_json(params, 'imagePath', filename)  # 修改json
                json.dump(new_params, filename[:-4] + '.json')  # 以图片名称 保存空标签
        progress_bar.close()
