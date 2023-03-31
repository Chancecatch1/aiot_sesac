#  강사님 정답
class AiStore:

    def __init__(self, name, s_id, locate):
        self.name = name
        self.s_id = s_id
        self.locate = locate
        self.count = 100
        self.price = 1000

    def set_product(self, count, price):
        self.count = count
        self.price = price

    def buy_item(self, count, amount):
        if count < self.count:
            if count * self.price < amount:
                self.count -= count
                changes = amount - count * self.price
                print('잔돈은 ' + str(changes))
            else:
                print('금액이 부족합니다.')
        else:
            print('재고가 부족합니다.')

    def get_name(self):
        return self.name

    def get_id(self):
        return self.s_id

    def get_locate(self):
        return self.locate

    def get_count(self):
        return self.count

    def get_price(self):
        return self.price


if __name__ == '__main__':

    s_name = input('스토어 지점이름 입력:')
    s_id = input('스토어 지점번호 입력:')
    locate = input('스토어 위치 입력:')
    store = AiStore(s_name, s_id, locate)
    print(store.get_name() + ' 스토어가 생성 되었습니다.')

    for i in range(10):

        print('스토어 조회는 1번, 구매는 2번, 상품 관리는 3번, 종료는 4번 을 눌러주세요')
        num = input()
        if num =='1':

            print('스토어 이름: ' + store.get_name())
            print('스토어 번호: ' + store.get_id())
            print('스토어 위치: ' + store.get_locate())
            print('재고 상황: ' + str(store.get_count()))
            print('상품 가격: ' + str(store.get_price()))

        elif num == '2':
            count = input('구매할 상품 개수 입력')
            count = int(count)
            need_price = count * store.get_price()
            print(str(need_price) + ' 금액을 입력해 주세요')
            amount = input()
            amount = int(amount)
            store.buy_item(count, amount)

        elif num == '3':
            count = input('상품 재고 입력')
            price = input('상품 가격 입력')
            store.set_product(int(count), int(price))

        elif num == '4':
            print('종료합니다')
            break
        else:
            print('잘못된 입력입니다')