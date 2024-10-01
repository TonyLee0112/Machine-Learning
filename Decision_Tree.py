# Logistic Regression Model 로 Multiclass Classification
import pandas as pd
import matplotlib.pyplot as plt

wine = pd.read_csv('https://bit.ly/wine_csv_data')
# print(wine.head()) # head 함수는 처음 5개의 원소들을 보여줌
# Red wine = Class 0, White wine = Class 1
# print(wine.info()) # 각 열의 Data type 과 누락된 데이터가 있는지 확인하는데 유용
# print(wine.describe()) # 열에 대한 간단한 통계 데이터 출력
data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data,target,test_size=0.2,random_state=42) # Train_Set 과 Test_Set Size 비율을 8:2 로 조정

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled,train_target)
# print(lr.score(train_scaled,train_target)) 0.780
# print(lr.score(test_scaled,test_target)) 0.777 -> 과소적합

# Decision Tree -> 모델의 판단 이유를 설명하기 쉽다.
# 데이터를 잘 나눌 수 있는 질문을 찾는다면 계속 질문을 추가해서 분류 정확도를 높일 수 있다.
# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(train_scaled,train_target)
# print(dt.score(train_scaled,train_target))
# print(dt.score(test_scaled,test_target))


# from sklearn.tree import plot_tree
# plt.figure(figsize=(10,7))
# plot_tree(dt,max_depth=1,filled=True, feature_names=['alcohol','sugar','pH'])
# plt.show()

# Gini Impurity = 1 - (음성 클래스 비율^2 + 양성 클래스 비율^2)
# DecisionTreeClassifier 는 이 값을 기준으로 각 노드에서 데이터를 분할하는 기준을 만들어냄.
# DT Model 은 Parent Node 와 Child Node 의 불순도 차이가 가능한 크도록 트리를 성장시킨다.
# 어떻게? 자식 노드의 불순도를 샘플 개수에 비례하여 모드 더하고 부모 노드의 불순도에서 뺀다. 이 차이 값을 Information Gain 이라고 한다.
# 부모의 불순도 - (왼쪽 노드로 간 비율 * 왼쪽 노드 불순도 + 오른쪽 노드로 간 비율 * 오른쪽 노드 불순도)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled,train_target)
# print(dt.score(train_scaled,train_target))
# print(dt.score(test_scaled,test_target))

# from sklearn.tree import plot_tree
# plt.figure(figsize=(20,15))
# plot_tree(dt,filled=True,feature_names=['alcohol','sugar','pH'])
# plt.show()

# DecisionTree 의 장점 : Data Preprocessing 을 할 필요가 없다.
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input,train_target)
# print(dt.score(train_input,train_target))
# print(dt.score(test_input,test_target))

# from sklearn.tree import plot_tree
# plt.figure(figsize=(20,15))
# plot_tree(dt,filled=True, feature_names=['alcohol','sugar','pH'])
# plt.show()

print(dt.feature_importances_)
