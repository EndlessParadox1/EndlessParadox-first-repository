import pickle
import random
from copy import deepcopy
import imageio.v2 as imageio
from PIL import Image
from matplotlib import pyplot as plt


# 读取图像
def GetImage(ori_image):
    img = Image.open(ori_image)
    color = []
    width, height = img.size
    for j in range(height):
        temp = []
        for i in range(width):
            r, g, b = img.getpixel((i, j))[:3]
            temp.append([r, g, b, r + g + b])
        color.append(temp)
    return color, img.size


# 初始化
def RandGroup(size, target):
    width, height = size
    group = []
    for i in range(100):
        individual = []
        for j in range(height):
            temp = []
            for k in range(width):
                t_r, t_g, t_b, t_a = target[j][k]
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                a = abs(t_r - r) + abs(t_g - g) + abs(t_b - b)
                temp.append([r, g, b, a])
            individual.append(temp)
        group.append([individual, 0])
    return group


# 计算适应度
def CalcFitness(group, target):
    for individual in group:
        count = 0
        for i, row in enumerate(individual[0]):
            for j, col in enumerate(row):
                t_r, t_g, t_b, t_a = target[i][j]
                r, g, b, a = col
                a = abs(t_r - r) + abs(t_g - g) + abs(t_b - b)
                count += a
        individual[1] = count
    group.sort(key=lambda x: x[1])


# 总体变异
def Variation(group, target):
    for individual in group:
        AltOffspring(individual, target)


# 个体变异
def SelfVariation(parent, t_r, t_g, t_b):
    # 较大变异，概率较小
    max_mutate_rate = 0.05
    mid_mutate_rate = 0.2
    # 较小变异，概率较大
    min_mutate_rate = 0.5
    offspring = deepcopy(parent)
    if random.random() < max_mutate_rate:
        offspring[0] = random.randint(0, 255)
    if random.random() < max_mutate_rate:
        offspring[1] = random.randint(0, 255)
    if random.random() < max_mutate_rate:
        offspring[2] = random.randint(0, 255)

    if random.random() < mid_mutate_rate:
        offspring[0] = min(max(0, offspring[0] + random.randint(-30, 30)), 255)
    if random.random() < mid_mutate_rate:
        offspring[1] = min(max(0, offspring[1] + random.randint(-30, 30)), 255)
    if random.random() < mid_mutate_rate:
        offspring[2] = min(max(0, offspring[2] + random.randint(-30, 30)), 255)

    if random.random() < min_mutate_rate:
        offspring[0] = min(max(0, offspring[0] + random.randint(-10, 10)), 255)
    if random.random() < min_mutate_rate:
        offspring[1] = min(max(0, offspring[1] + random.randint(-10, 10)), 255)
    if random.random() < min_mutate_rate:
        offspring[2] = min(max(0, offspring[2] + random.randint(-10, 10)), 255)

    offspring[3] = abs(offspring[2] - t_b) + abs(offspring[1] - t_g) + abs(offspring[0] - t_r)
    return offspring


# 替代
def AltOffspring(individual, target):
    for i, row in enumerate(individual[0]):
        for j, parent in enumerate(row):
            p_r, p_g, p_b, p_a = parent
            t_r, t_g, t_b, t_a = target[i][j]
            offsprings = []
            for k in range(5):
                offsprings.append(SelfVariation(parent, t_r, t_g, t_b))
            offsprings.sort(key=lambda x: x[3])
            if offsprings[0][3] < p_a:
                individual[0][i][j] = offsprings[0]


# 交叉
def Merge(individual1, individual2, size):
    width, height = size
    y = random.randint(0, height - 1)
    new_1 = [deepcopy(individual1[0][:y]) + deepcopy(individual2[0][y:]), 0]
    new_2 = [deepcopy(individual2[0][:y]) + deepcopy(individual1[0][y:]), 0]
    return new_1, new_2


# 选择
def Select(group, size):
    seek = len(group) // 2
    i = 0
    j = seek
    # 将后1/2的基因替换为前1/2基因的两两交叉
    while i < seek:
        group[j], group[j + 1] = Merge(group[i], group[i + 1], size)
        j += 2
        i += 2


# 保存生成的图片
def SavePic(group, generation, size):
    for k, individual in enumerate(group):
        img = Image.new('RGB', size)
        for j, row in enumerate(individual[0]):
            for i, col in enumerate(row):
                r, g, b = col[:3]
                img.putpixel((i, j), (r, g, b))
        if k == 0:
            img.save('best/第{}代.png'.format(str(generation)))
        elif k == 1:
            img.save('mid/第{}代.png'.format(str(generation)))
        else:
            img.save('worst/第{}代.png'.format(str(generation)))


# 保存坐标数据
def SavePlotData(group, generation, plotdata_):
    fitnessSum = 0
    for i in range(100):
        fitnessSum += group[i][1]
    plotdata_[0].append(group[0][1])
    plotdata_[1].append(group[99][1])
    plotdata_[2].append(fitnessSum / 100)
    plotdata_[3].append(generation)


# 备份
def SaveData(data, backup_):
    with open(backup_, 'wb') as f:
        pickle.dump(data, f)
    f.close()


# 读取备份
def ReadData(backup_):
    print('[INFO]: Read data from {}...'.format(backup_))
    with open(backup_, 'rb') as f:
        data = pickle.load(f)
        group = data['group']
        generation = data['generation']
    f.close()
    return group, generation


# 获取图片路径
def GetPaths(path):
    return ['{}/第{}代.png'.format(path, str(i)) for i in range(1, 51)]


# 运行
def run(ori_image, backup_, plotdata_, resume=False):
    data, size = GetImage(ori_image)
    if resume:
        group, generation = ReadData(backup_)
    else:
        group = RandGroup(size, data)
        generation = 0
    for _ in range(50):
        generation += 1
        Variation(group, data)
        CalcFitness(group, data)
        Select(group, size)
        CalcFitness(group, data)
        SaveData({'group': group, 'generation': generation}, backup_)
        SavePic((group[0], group[49], group[99]), generation, size)
        SavePlotData(group, generation, plotdata_)
        print('<generation>: {}, <Select>: {} {} {}'.format(generation, group[0][1], group[49][1],
                                                            group[99][1]))
    plt.plot(plotdata_[3], plotdata_[0], color='red', label='best')
    plt.plot(plotdata_[3], plotdata_[1], color='green', label='worst')
    plt.plot(plotdata_[3], plotdata_[2], color='blue', linestyle='--', label='average')
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.savefig('procedure.png')
    imageio.mimsave('best//evolution.gif', [imageio.imread(i) for i in GetPaths('best')], 'GIF', duration=0.1, loop=0)


if __name__ == '__main__':
    # 备份
    backup = 'backup.tmp'
    # 原始图像
    ori_img = 'test.png'
    # 折线图数据[优，差，均，代]
    plotdata = [[], [], [], []]
    # resume为True则读取备份文件，在其基础上进行自然选择和变异
    run(ori_img, backup, plotdata, False)
