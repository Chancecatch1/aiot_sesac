import json
import copy
import sys

with open('bank4.json', 'r') as f:
    bank = json.load(f)
    print(bank)


class Customer:

    def __init__(self, customer):
        self.c_id = customer['c_id']
        self.c_name = customer['name']
        self.accounts_num = customer['account_num']
        self.rat = customer['rat']
        self.total_amount = customer['total_amount']
        self.accounts = customer['account']

    def add_amount(self, a_id, amount):
        if a_id in self.accounts:
            self.accounts[a_id]['amount'] += amount
            self.total_amount = self.get_total_amount()
            self.rat = self.get_rat()
        else:
            print('해당 계좌가 없습니다')

    def sub_amount(self, a_id, amount):
        if a_id in self.accounts:
            if self.accounts[a_id]['amount'] > amount:
                self.accounts[a_id]['amount'] -= amount
                self.total_amount = self.get_total_amount()
                self.rat = self.get_rat()
            else:
                print('계좌의 금액이 부족합니다.')
        else:
            print('해당 계좌가 없습니다.')

    def add_account(self, a_id):
        if a_id in self.accounts:
            print('해당 계좌가 이미 존재합니다')
        else:
            self.accounts[a_id] = {'a_id': a_id, 'amount': 0, 'c_id': self.c_id}
            self.accounts_num += 1

    def get_total_amount(self,):
        t_amount = 0
        for a_id in self.accounts:
            t_amount += self.accounts[a_id]['amount']
        return t_amount

    def get_rat(self):
        total_amount = self.get_total_amount()
        if total_amount > 100000:
            rat = 'vvip'
        elif total_amount > 10000:
            rat = 'vip'
        elif total_amount > 1000:
            rat = 'gold'
        elif total_amount > 100:
            rat = 'silver'
        else:
            rat = 'bronze'

        return rat

    def update(self, bank):
        bank[self.c_id] = self.get_customer()
        # for a_id in self.accounts.index:
        #     a_df.loc[a_id] = self.accounts.loc[a_id]

    def get_name(self):
        return self.c_name

    def get_cid(self):
        return self.c_id

    def get_accounts(self):
        return self.accounts

    def get_customer(self):
        return {'c_id': self.c_id, 'name': self.c_name,
                'account_num': self.accounts_num,
                'total_amount': self.total_amount,
                'rat': self.rat,
                'account': self.accounts
                }


def create_customer(c_id , c_name):

    customer = {'c_id': c_id, 'name': c_name,
                'account_num': 0,
                'total_amount': 0,
                'rat': 'normal',
                'account': {}
                }

    bank[c_id] = customer


def show_list():
    result = []
    for c_id in bank:
        c = bank[c_id]
        result.append([c['name'],c['c_id'],c['rat'],c['total_amount']])
    print(result)
    return result

def search_customer(c_id):
    print(bank)
    if c_id in bank:
        customer = copy.copy(bank[c_id])
        return Customer(customer)
    else:
        print('고객 번호를 찾지 못했습니다.')
        return None


def show_customer(c_id):

    customer = search_customer(c_id)
    if customer is None:
        return None
    customer_view = [[customer.get_name(),
                     customer.get_cid(),
                     customer.get_rat(),
                     customer.get_total_amount()]]

    accounts = customer.get_accounts()
    account_view = []
    for a_num in accounts:
        account_view.append(accounts[a_num])

    return customer_view, account_view

def update(customer):
    bank[customer.get_cid()] = customer.get_customer()

def group_rat_count():
    group_rat = {}
    for c in bank:
        rat = bank[c]['rat']
        # group_rat[rat] += 1
        group_rat[rat] = group_rat.get(rat, 0) + 1

    print(group_rat)


