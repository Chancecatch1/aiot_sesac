while True:
    print('가위바위보 게임 입니다. 가위:1, 바위:2, 보:3 ')
    rsp1 = input('a 유저의 번호를 선택해주세요')
    rsp2 = input('b 유저의 번호를 선택해주세요')
# 2가 1을 이기고, 3이 2를 이긴다. 1이 3을 이긴다.
    # a가 승리일시 'a 승리' 출력
    # b가 승리일시 'b 승리' 출력
    # a와 b가 무승부일시 '무승부' 출력
    if rsp1 == rsp2 :
        print('무승부')
    elif rsp1 == '1':
        if rsp2 == '2':
            print('b 승리')
        else : print('a 승리')
    elif rsp1 == '2':
        if rsp2 == '1':
            print('a 승리')
        else : print('b 승리')
    elif rsp1 == '3':
        if rsp2 == '1':
            print('b 승리')
        else : print('a 승리')
    else :
        print('잘못 입력했습니다.')