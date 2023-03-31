from flask import Flask, request, render_template, redirect, url_for
import sys

# Flask 인스턴스 생성
app = Flask(__name__) # 던더네임 : 현재 활성 모듈의 이름
from bank import *
from wtforms import Form, StringField, PasswordField, TextAreaField, validators
# wtforms는 웹으로 전달하는 form 데이터의 형식을 정의함으로써 폼 데이터 양식을 관리한다.

class CaForm(Form):
    c_id = StringField('고객 아이디: ', [validators.length(max=20)])
    a_num = StringField('계좌 번호: ', [validators.length(max=20)])

# post : (create) 데이터를 생성하거나 업데이트 할 때
# get : (select) 서버의 리소스에서 데이터를 요청할 때
# 웹표현 : route() 메소드 사용
# 맨앞에 @가 붙는 것은 장식자(decorator)를 나타낸다, 장식자가 url연결에 활용
@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        op = request.form['option']
        if op == '0':
            return redirect(url_for('ccpage'))
        elif op == '1':
            return redirect(url_for('capage'))
        elif op == '2':
            return redirect(url_for('dwpage', c_id = 'nan', a_num = 'nan'))
        elif op == '3':
            return redirect(url_for('viewpage'))

    return render_template('index.html')

@app.route("/ccpage", methods=['POST', 'GET'])
def ccpage():
    if request.method == 'POST':
        c_id = request.form['cId']
        c_name = request.form['cName']
        create_customer(c_id, c_name)

        return redirect('/')

    return render_template('ccpage.html')


@app.route("/capage", methods=['POST', 'GET'])
def capage():
    form = CaForm(request.form)
    if request.method == 'POST' and form.validate():
        c_id = form.c_id.data
        a_num = form.a_num.data
        customer = search_customer(c_id)
        customer.add_account(a_num)
        update(customer)
        return redirect('/')

    return render_template('capage.html', form=form)

@app.route("/dwpage/<c_id>/<a_num>", methods=['POST', 'GET'])
def dwpage(c_id ='nan', a_num = 'nan'):
    if request.method == 'POST':
        # 고객 아이디, 계좌번호, 입출금 옵션, 입력금액 요청 폼에서 받아오기
        # 고객 아이디로 고객인스턴스 생성
        # 입출금 옵션이 입금이면 고객인스턴스에서 입금함수 실행 아니면 출금함수 실행
        # 데이터 업데이트
        c_id = request.form['cId']
        a_num = request.form['aNum']
        amount = int(request.form['amount'])
        customer = search_customer(c_id)

        if request.form['option'] == '0':
            customer.add_amount(a_num, amount)
        elif request.form['option'] == '1':
            customer.sub_amount(a_num, amount)

        update(customer)

        return redirect(url_for('viewpage'))

    return render_template('dwpage.html', c_id=c_id, a_num=a_num)


@app.route("/viewpage", methods=['POST', 'GET'])
def viewpage():
    if request.method == 'POST':
        c_id = request.form['inputCId']
        customer_view, accounts_view = show_customer(c_id)

        return render_template('viewpage.html',
                               customers=customer_view,
                               accounts=accounts_view
                               )

    return render_template('viewpage.html',
                           customers=show_list(),
                           accounts=None
                           )

if __name__ == '__main__':
    app.run('0.0.0.0',9999, debug=True)