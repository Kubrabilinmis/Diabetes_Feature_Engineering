"""
                                                                Feature  Engineering
İş Problemi:
Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

Veri Seti Hikayesi :
Veriseti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.ABD'deki Arizona Eyaleti'nin en büyük 5.şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. Hedef değişken "Outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir

Pregnancies    : Hamilelik sayısı
Glucose        : Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
Blood Pressure : Kan Basıncı (Küçüktansiyon) (mm Hg)
SkinThickness  : Cilt Kalınlığı
Insulin        : 2 saatlik serum insülini (mu U/ml)
Diabetes Pedigree Function  : Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
BMI            : Vücut kitle endeksi
Age            : Yaş (yıl)
Outcome        : Hastalığa sahip(1) ya da değil (0)

"""


# Görev 1 : Keşifçi Veri Analizi
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

from helpers.eda import *
from helpers.data_prep import *

def load():
    data = pd.read_csv("datasets/diabetes.csv")
    return data

df = load()
df.columns = [col.upper() for col in df.columns]


# Adım 1:Genel resmi inceleyiniz
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
# burada pregnants ve age  dısında dıger degıskenlerın mın degerı 0 ????


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# 1 kategorık  , 8 sayısal


# Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
cat_summary(df, "OUTCOME")
# dıabet hastalıgına sahıp 268 kısı var oranı yüzde 34 , olmayan ıse 500 kısı var oranı yaklasık yuzde 65 gozukmektedır

for col in num_cols:
    num_summary(df, col)
# num_cols analiz


# Adım 4:Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

for col in num_cols:    # target acısından sayısal degıskenlerı analız et dıyoruz
    target_summary_with_num(df, "OUTCOME", col)
# hamilelik ilişkisinin diyabetle ilgisi varmıs gibi gözüküyor
# glikoz test sonucu pozitif cıkanlarda daha yüksek
# kan basıncı dıabet olanlarda daha fazla ımıs kayde deger mı bılmıyoruz
# insulin ort 100 mus kayde deger mı bilmiyoruz
# Age de  diyabet olan ve olmayanlarda ort degerler yakın gibi duruyor ama bılmıyoruz



# Adım 5: Aykırı gözlem analizi yapınız.
for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 6: Eksik gözlem analizi yapınız.
df.isnull().sum()
# eksik gözlem bulunmamakta


# Adım 7: Korelasyon analizi yapınız.
f, ax = plt.subplots(figsize=[7, 5])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="YlGnBu")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()




# Görev 2 : Feature Engineering
# Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.

df[["GLUCOSE","BLOODPRESSURE","SKINTHICKNESS","INSULIN","BMI"]]= df[["GLUCOSE","BLOODPRESSURE","SKINTHICKNESS","INSULIN","BMI"]].replace(0,np.NaN)

missing_values_table(df, True)

na_cols = missing_values_table(df, True)
# eksık degere sahıp tum degıskenler geldı.

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "OUTCOME", na_cols)


def median_target(variable):
    temp = df[df[variable].notnull()]
    temp = temp[[variable, 'OUTCOME']].groupby(['OUTCOME'])[[variable]].median().reset_index()
    return temp


columns = df.columns
columns = columns.drop("OUTCOME")

for col in columns:
    df.loc[(df['OUTCOME'] == 0) & (df[col].isnull()), col] = median_target(col)[col][0]
    df.loc[(df['OUTCOME'] == 1) & (df[col].isnull()), col] = median_target(col)[col][1]

# tek degıskenlı bakalım..
for col in num_cols:
    print(col, check_outlier(df, col))


# cok degıskenlı bırde bakalım..
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df)    # local outlıer factor skorları gelır
df_scores = clf.negative_outlier_factor_
df_scores[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

th = np.sort(df_scores)[5]
df[df_scores < th]
df[df_scores < th].shape # (5, 9)
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T    # sebeplere bakalım

df[df_scores < th ].index
df.drop(axis=0, labels=df[df_scores < th ].index)
df = df.drop(axis=0, labels=df[df_scores < th].index)
df.head()
df.shape


# Adım 2: Yeni değişkenler oluşturunuz.
# BMI  : Vücut kitle endeksi
df["BMI"].min()
df["BMI"].max()
df["NEW_BMI"] = pd.cut(x=df["BMI"], bins = [0,18.5,24.9,29.9,100], labels = [ "Underweight","Healthy","Overweight","Obese"])


# GLUCOSE  Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
df["NEW_GLUCOSE"] = pd.cut(x=df["GLUCOSE"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])


# Age  : Yaş (yıl)
df["AGE"].min()  # 21
df["AGE"].max()  # 55
df.loc[(df['AGE'] <= 30), "NEW_AGE"] = "young"
df.loc[(df['AGE'] > 30) & (df["AGE"] <= 50), "NEW_AGE"] = "middle_age"
df.loc[(df['AGE'] > 50), "NEW_AGE"] = "old"
df.head()

df["NEW_AGE"].value_counts()

# BloodPressure  : Kan Basıncı
df.loc[(df['BLOODPRESSURE'] < 70), 'NEW_BLOOD_CAT'] = "hipotansiyon"
df.loc[(df['BLOODPRESSURE'] >= 70) & (df['BLOODPRESSURE'] < 90), 'NEW_BLOOD_CAT'] = "normal"
df.loc[(df['BLOODPRESSURE'] >= 90), 'NEW_BLOOD_CAT'] = "hipertansiyon"

#Insulin        : 2 saatlik serum insülini (mu U/ml)
df["NEW_INSULIN"] = pd.cut(x=df["INSULIN"],
                           bins=[0, 140, 200, df["INSULIN"].max()],
                           labels=["Normal", "Hidden_diabetes", "Diabetes"])



# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.head()


# Adım 5: Model oluşturunuz.
y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
#  0.8995


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
plot_importance(rf_model, X_train)










