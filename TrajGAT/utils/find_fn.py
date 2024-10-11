import os
import re


def _find_best_weight_file(city, lorentz, sim):
    # 构建文件模式
    pattern = f"{city}_{lorentz}_TrajGAT_{sim}_wts_hr10 (.+).pkl"

    best_file = None
    best_value = float('-inf')

    # 遍历目录中的所有文件
    for filename in os.listdir('./model/wts/'):
        match = re.match(pattern, filename)
        if match:
            # 提取第五个参数的值
            value = float(match.group(1))
            if value > best_value:
                best_value = value
                best_file = './model/wts/' + filename

    return best_file, best_value

def find_best_weight_file(city, lorentz, sim):
    if lorentz == 1 or lorentz == 1.0:
        fn1 ,v1 = _find_best_weight_file(city, '1', sim)
        fn2, v2 = _find_best_weight_file(city, '1.0', sim)
        if v2 > v1:
            return fn2
        else:
            return fn1
    elif lorentz == 0 or lorentz == 0.0:
        fn1, v1 = _find_best_weight_file(city, '0', sim)
        fn2, v2 = _find_best_weight_file(city, '0.0', sim)
        if v2 > v1:
            return fn1
        else:
            return fn2
    else:
        raise ValueError(f"Invalid lorentz value: {lorentz}")


if __name__ == '__main__':

    # 使用示例
    model_best_wts_path = '../model/wts/'
    param1, param2, param3, param4 = 'chengdu', 1, 'TrajGAT', 'dtw'
    best_file = find_best_weight_file(param1, param2, param4)

    print(f"The best weight file is: {best_file}")




