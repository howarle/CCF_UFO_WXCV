def convert_X_1(arch_str):
    '''
    arch_str: input arch
    return: 
    '''
    temp_arch = []
    total_1 = 0
    total_2 = 0
    ts = ''
    for i in range(len(arch_str)):
        if i % 3 != 0:
            elm = arch_str[i]
            ts = ts + elm
            if elm == '1':
                temp_arch = temp_arch + [1]
            elif elm == '2':
                temp_arch = temp_arch + [2]
            elif elm == '3':
                temp_arch = temp_arch + [3]
            else:
                temp_arch = temp_arch + [0]

    return temp_arch

def convert_X(arch_str):
    '''
    arch_str: input arch
    return: 
    '''
    temp_arch = []
    total_1 = 0
    total_2 = 0
    ts = ''
    for i in range(len(arch_str)):
        if i % 3 != 0 and i != 0 and i <= 30:
            elm = arch_str[i]
            ts = ts + elm
            if elm == 'l' or elm == '1':
                temp_arch = temp_arch + [1, 1, 0, 0]
            elif elm == 'j' or elm == '2':
                temp_arch = temp_arch + [0, 1, 1, 0]
            elif elm == 'k' or elm == '3':
                temp_arch = temp_arch + [0, 0, 1, 1]
            else:
                temp_arch = temp_arch + [0, 0, 0, 0]

        elif i % 3 != 0 and i != 0 and i > 30:
            elm = arch_str[i]
            if elm == 'l' or elm == '1':
                temp_arch = temp_arch + [1, 1, 0, 0, 0]
            elif elm == 'j' or elm == '2':
                temp_arch = temp_arch + [0, 1, 1, 0, 0]
            elif elm == 'k' or elm == '3':
                temp_arch = temp_arch + [0, 0, 1, 1, 0]
            else:
                temp_arch = temp_arch + [0, 0, 0, 0, 1]
    return temp_arch