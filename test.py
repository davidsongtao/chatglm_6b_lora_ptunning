import traceback

try:
    example = '你好么'
    a = 1/0
    print(example)
except:
    print(f'{example} -> \n{traceback.format_exc()}')
